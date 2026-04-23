import torch
import torch.nn as nn


def init_prototypes(x: torch.Tensor, y: torch.Tensor, n_classes: int):
    prototypes, proto_labels = [], []
    for c in range(n_classes):
        idx = (y == c).nonzero(as_tuple=True)[0]
        prototypes.append(x[idx[0]].clone())
        proto_labels.append(c)
    return torch.stack(prototypes), torch.tensor(proto_labels)


class GMLVQLoss(nn.Module):
    """GMLVQ cost: sigmoid of relative distance margin mu = (d_c - d_w) / (d_c + d_w)."""
    def forward(self, d_c: torch.Tensor, d_w: torch.Tensor) -> torch.Tensor:
        mu = (d_c - d_w) / (d_c + d_w + 1e-8)
        return torch.sigmoid(mu)


class GMLVQ(nn.Module):

    def __init__(self, n_features: int, n_classes: int, lr_w=0.01, lr_r=0.01, epochs=20):
        super().__init__()
        self.epochs  = epochs
        self.n_classes = n_classes
        self.lr_w    = lr_w
        self.lr_r    = lr_r
        self.eps     = 1e-8

        self.W = nn.Parameter(torch.empty(n_classes, n_features))
        self.r = nn.Parameter(torch.ones(n_features))
        nn.init.normal_(self.W, mean=0.0, std=0.1)

        self.criterion  = GMLVQLoss()
        self.W_labels: torch.Tensor | None = None

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        W_init, self.W_labels = init_prototypes(X, y, self.n_classes)
        with torch.no_grad():
            self.W.copy_(W_init)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for i in range(len(X)):
                x, label = X[i], y[i].item()

                with torch.no_grad():
                    r = self.r.clamp(min=1e-6)

                    # find closest correct / wrong prototype
                    d_correct, d_wrong = [], []
                    for j in range(len(self.W)):
                        diff = x - self.W[j]
                        d = (r * diff * diff).sum().item()
                        if self.W_labels[j].item() == label:
                            d_correct.append((d, j))
                        else:
                            d_wrong.append((d, j))

                    wc = min(d_correct, key=lambda t: t[0])[1]
                    ww = min(d_wrong,   key=lambda t: t[0])[1]

                    w_c = self.W[wc]
                    w_w = self.W[ww]

                    # GMLVQ cost signal
                    diff_c = x - w_c
                    diff_w = x - w_w
                    d_c = (r * diff_c * diff_c).sum()
                    d_w = (r * diff_w * diff_w).sum()

                    phi = self.criterion(d_c, d_w)

                    # gradient updates per GMLVQ rules
                    self.W[wc] += self.lr_w * 2 * r * diff_c * phi
                    self.W[ww] -= self.lr_w * 2 * r * diff_w * phi
                    self.r     += self.lr_r * (diff_w ** 2 * phi - diff_c ** 2 * phi)
                    self.r.clamp_(min=1e-6)
                    # project r onto unit simplex — prevents r from growing
                    # unboundedly, which collapses mu → 0 and phi → 0.5
                    self.r /= self.r.sum()

                    epoch_loss += phi.item()

            print(f"Epoch {epoch + 1}/{self.epochs}  loss={epoch_loss / len(X):.4f}")

    def _distances(self, x: torch.Tensor) -> torch.Tensor:
        r = self.r.clamp(min=1e-6)
        diff = x.unsqueeze(0) - self.W
        return (r * diff * diff).sum(dim=-1)

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        preds = []
        for x in X:
            dists = self._distances(x)
            preds.append(self.W_labels[dists.argmin()])
        return torch.stack(preds)

    @torch.no_grad()
    def score(self, X: torch.Tensor, y: torch.Tensor) -> float:
        return (self.predict(X) == y).float().mean().item()

    def get_relevance(self) -> torch.Tensor:
        return self.r.detach()


if __name__ == '__main__':
    import os
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    os.system('clear')

    X_np, y_np = load_iris(return_X_y=True)
    X_np = StandardScaler().fit_transform(X_np)
    x_tr, x_te, y_tr, y_te = train_test_split(X_np, y_np, test_size=0.2, random_state=0)

    x_tr = torch.tensor(x_tr, dtype=torch.float32)
    x_te = torch.tensor(x_te, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    y_te = torch.tensor(y_te, dtype=torch.long)

    n_features, n_classes = x_tr.shape[1], len(y_tr.unique())
    model = GMLVQ(n_features=n_features, n_classes=n_classes, lr_w=0.005, lr_r=0.005, epochs=100)
    model.fit(x_tr, y_tr)

    print(
        f"Train: {model.score(x_tr, y_tr):.3f}",
        f"Test:  {model.score(x_te, y_te):.3f}",
        sep="\t|\t",
    )
