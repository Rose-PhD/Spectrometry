import numpy as np

def init_protoytypes(x, y, n_classes):
    """Computes x-prototype and unique y labels"""
    prototypes = []
    proto_labels = []

    for c in range(n_classes):
        idx = np.where(y == c)[0]
        x_prototype = x[idx[0]].copy()

        prototypes.append(x_prototype)
        proto_labels.append(c)
    return np.array(prototypes), np.array(proto_labels)


def distance(x, w, omega):
    """Computes the distacne of point in new vector space"""
    diff = x - w
    return np.sum(omega * diff * diff)


class GMLVQ:

    def __init__(self, lr_w=0.01, lr_r = 0.01, epochs=20, EPS=1e-8):
        self.lr_w = lr_w
        self.lr_r = lr_r
        self.epochs = epochs
        self.eps = EPS

    
    def fit(self, X, y):
        n_classes = len(np.unique(y))
        self.W, self.W_labels = init_protoytypes(X, y, n_classes)
        self.r = np.ones(X.shape[-1])

        for epoch in range(self.epochs):
            for i in range(len(X)):
                x = X[i]
                label = y[i]

                d_correct, d_wrong = [], []
                for j, w in enumerate(self.W):
                    d = distance(x, w, self.r)

                    if self.W_labels[j] == label:
                        d_correct.append((d, j))
                    else:
                        d_wrong.append((d, j))
                
                wc = min(d_correct, key=lambda t: t[0])[1]
                ww = min(d_wrong, key=lambda t: t[0])[1]

                w_c = self.W[wc]
                w_w = self.W[ww]

                # compute GMLVQ cost signal
                d_c = distance(x, w_c, self.r)
                d_w = distance(x, w_w, self.r)
                mu = (d_c - d_w) / (d_c + d_w + self.eps)

                # sigmoid cost for update
                phi = 1 / (1 + np.exp(-mu))

                diff_c = x - w_c
                diff_w = x - w_w
                
                grad_c = 2 * self.r * diff_c * phi
                grad_w = 2 * self.r * diff_w * phi

                # parameter update of parameters
                self.W[wc]  += self.lr_w * grad_c
                self.W[ww] -= self.lr_w * grad_w

                # update for the relevance vector r
                self.r += self.lr_r * (diff_w ** 2 * phi - diff_c ** 2 * phi)
                self.r = np.clip(self.r , 1e-6, None)

    
    def predict(self, X):
        preds = []
        for x in X:
            distances = [distance(x, w, self.r) for w in self.W]
            preds.append(self.W_labels[np.argmin(distances)])
        return np.array(preds)
    
    def get_relevance(self):
        return self.r
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    

if __name__ =='__main__':
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import os

    os.system('clear')


    X, y = load_iris(return_X_y=True)
    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = GMLVQ(lr_r=0.005, epochs=100)
    model.fit(x_train, y_train)

    model.score(x_test, y_test)

    print(f'Model Train: {model.score(x_train, y_train):.3f}',f'Model Test prediction: {model.score(x_test, y_test):.3f}', sep='\t|\t')
    
    
