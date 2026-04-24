import torch
import torch.nn as nn

torch.manual_seed(0)


def initiate_prototypes(X: torch.Tensor, y: torch.Tensor, n_classes: int):
    """Initaliazation of prototypes"""
    prototypes = []
    proto_labels = []
    for c in range(n_classes):
        class_mask = (y == c)
        proto = X[class_mask][0]
        prototypes.append(proto)
        proto_labels.append(c)
    prototypes = torch.stack(prototypes)
    proto_labels = torch.tensor(proto_labels, dtype=torch.float32)
    return prototypes, proto_labels


def tensor_split(in_tensor: torch.Tensor, target_tensor, ratio: float =0.2):
    """Partitions tensor based on test_size"""
    minor_ratio = int(ratio * len(in_tensor))
    major_ratio = len(in_tensor) - minor_ratio
    major_data, minor_data = random_split(
        in_tensor, 
        [major_ratio, minor_ratio]
    )
    return  in_tensor[major_data.indices], in_tensor[minor_data.indices],\
          target_tensor[major_data.indices], target_tensor[minor_data.indices]


class GMLVQLoss(nn.Module):
    """Computes the GMLVQ Loss and applies a non linear activation"""
    def forward(self, d_correct, d_wrong, EPS=1e-8):
        mu = (d_correct - d_wrong) / (d_correct + d_wrong + EPS)
        return torch.relu(mu)
    

class GMLVQ(nn.Module):
    """Definition of GMLVQ Model for relevance matrix learning"""

    def __init__(self, in_features: int, n_classes: int, lr_w=0.01, lr_r=0.01, epochs=10):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_classes, in_features))
        self.r = nn.Parameter(torch.ones(in_features))
        self.lr_w = lr_w
        self.lr_r = lr_r
        self.criterion = GMLVQLoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.w, 'lr': self.lr_w},
            {'params': self.r, 'lr': self.lr_r}
        ])
        nn.init.normal_(self.w, mean=0, std=0.1)
        self.epochs = epochs

    def _distance(self, x: torch.Tensor):
        """Computes the distance of separation in new d space"""
        r = torch.softmax(self.r, dim=0)
        diff = x.unsqueeze(0) - self.w
        return (r * diff * diff).sum(dim=1)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Computes the relevance matrix through iterative algorithms"""
        n_classes = len(y.unique())
        prototypes, proto_labels = initiate_prototypes(X, y, n_classes)
        self.w = nn.Parameter(prototypes)
        self.register_buffer('w_labels', proto_labels)

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for i in range(len(X)):
                x, label = X[i], y[i]
                distances = self._distance(x)
    
                correct_mask = label == self.w_labels
                wrong_mask = label != self.w_labels

                d_correct = distances[correct_mask].min()
                d_wrong = distances[wrong_mask].min()

                loss = self.criterion(d_correct, d_wrong)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch: {epoch + 1} / {self.epochs}', f'loss: {epoch_loss / len(X):.4f}', sep='\t|')
    
    @torch.no_grad()
    def predict(self, X: torch.Tensor, y: torch.Tensor):
        """Computes the predictions on X"""
        preds = []
        for x in X:
            distances = self._distance(x)
            class_pred = self.w_labels[distances.argmin()]
            preds.append(class_pred)
        return torch.stack(preds)
    
    @torch.no_grad()
    def score(self, X: torch.Tensor, y: torch.Tensor):
        """Computes the mean square score"""
        return (self.predict(X, y) == y).float().mean().item()


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from torch.utils.data import random_split
    import os

    X, y = load_iris(return_X_y=True)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    os.system('clear')
    
    x_train, x_test, y_train, y_test = tensor_split(X, y, ratio=0.1)
    n_classes = len(y_train.unique())
    in_features = x_train.shape[-1]

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)

    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    

    model = GMLVQ(in_features, n_classes, epochs=100)
    model.fit(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print(f'Test score: {test_score}')







