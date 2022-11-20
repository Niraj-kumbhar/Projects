class SimpleLR:
    """
    Simple Linear Regression based on OLS method
    -------------------------
    fit(X_train.values, y_train_values) --> for training the model
    predict(x_test.values) --> prediction function
    parameters() --> print slope and b value 
    """
    def __init__(self) -> None:
        self.w = None
        self.b = None

    def parametres(self):
        print(f"w/coef_:{self.w}\nb/y_intercept_:{self.b}")

    def fit(self, X_train, y_train):
        num = 0
        den = 0

        for i in range(X_train.shape[0]):
            num += (X_train[i]-X_train.mean()) * (y_train[i]-y_train.mean())
            den += (X_train[i] - X_train.mean())**2

        self.w = num/den
        self.b = y_train.mean() - (self.w * X_train.mean())
        print("Model trained")

    def predict(self, X_test):
        return (self.w*X_test) + self.b

class LR:
    """
    Multiple Linear Regression based on OLS method
    -------------------------
    fit(X_train.values, y_train_values) --> for training the model
    predict(x_test.values) --> prediction function
    parameters() --> print slope and b value 
    """

    def __init__(self) -> None:
        self.coef_ = None
        self.intercept_ = None

    def parametres(self):
        print(f"w/coef_:{self.coef_}\nb/y_intercept_:{self.intercept_}")

    def fit(self,X_train, y_train):
        import numpy as np
        X_train = np.insert(X_train,0,1,axis=1)

        #calculating inverse
        betas = np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X_test):
        import numpy as np
        y_pred = np.dot(X_test,self.coef_) + self.intercept_
        return y_pred
        


