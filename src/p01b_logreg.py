import numpy as np
import util

from linear_model import LinearModel


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        # theta 0으로 초기화 조건 1번조건 
        self.theta = np.zeros(n) 
        
        epsilon = self.eps

        for i in range(self.max_iter):
            h = self.sigmoid(x @ self.theta)
            
            grad = x.T @ (h - y) / m

            hessian = (x.T * (h * (1-h)) @ x) / m
            
            # newton's method 업데이트 (np.linalg.inv)
            new_theta = self.theta - np.linalg.solve(hessian, grad)
            
            # theta 변화량이 1e-5 보다 작아질때까지만 - 2번조건
            if np.linalg.norm(new_theta - self.theta, 1) < epsilon:
                break
            self.theta = new_theta

            if self.verbose:
                loss = self.compute_loss(x, y)
                print(f'Iteration {i+1}, Loss : {loss}')

        print(self.theta)
        # *** END CODE HERE ***

    def predict(self, x):
        
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return (self.sigmoid(x @ self.theta) >= 0.5).astype(int)
        # *** END CODE HERE ***


    # sigmoid 함수 정의
    def sigmoid(self, z):
        
        # overflow 방지
        z = np.clip(z, -709, 709)
        return 1 / (1 + np.exp(-z))
    
    # loss 계산
    def compute_loss(self, x, y):
        m = len(y)
        h = self.sigmoid(x @ self.theta)
        # logistic regression 의 cost function 을 통해 cost 반환
        return -1 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))


def main(train_path, eval_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load Data
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # training set 의 결정경계 Plot
    util.plot(x_train, y_train, clf.theta, save_path='code/src/res/p1_train_plot.png')
    print("Theta is :", clf.theta)
    print("Accuracy on training set : ", np.mean(clf.predict(x_train) == y_train))
    
    # validation set 의 결정경계 Plot
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    util.plot(x_valid, y_valid, clf.theta, save_path='code/src/res/p1_valid_plot.png')
    print("Accuracy on validation set : ", np.mean(clf.predict(x_valid) == y_valid).astype(float))

if __name__ == "__main__":
    train_path = 'code/data/ds1_train.csv'
    eval_path = 'code/data/ds1_valid.csv'

    main(train_path, eval_path)