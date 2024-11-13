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
        # theta 0으로 초기화 조건 - feature 수만큼 
        self.theta = np.zeros(n) 
        
        epsilon = self.eps

        for _ in range(self.max_iter):
            h = self.sigmoid(x @ self.theta)
            
            grad = x.T @ (y - h) / m

            hessian = (x.T @ (h * (1-h)) @ x) / m

            new_theta = self.theta - np.linalg.inv(hessian) @ grad

            if np.linalg.norm(new_theta - self.theta, 1) < epsilon:
                break
            self.theta = new_theta

            if self.verbose:
                loss = self.compute_loss(x, y)
                print(f'Loss: {loss}')

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
        return 1 / (1 + np.exp(-z))
    
    # loss 계산
    def compute_loss(self, x, y):
        m = len(y)
        h = self.sigmoid(x @ self.theta)
        # logistic regression 의 cost function 을 통해 cost 반환
        return -1 / m * (y.T @ np.log(h) + (1 - y).T @ np.log(1-h))


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    print("Loading training data")
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LogisticRegression()
    print("training model")
    clf.fit(x_train, y_train)

    util.plot(x_train, y_train, clf.theta, save_path='/Users/kimtaewan/Desktop/개인공부/ML_440/code/src/res')
    x_eval, _ = util.load_dataset(eval_path, add_intercept=True)
    predictions = clf.predict(x_eval)

    np.savetxt(pred_path, predictions, fmt='%d')
    # *** END CODE HERE ***
    #     

    if __name__ == "__main__":
        train_path = 'code/data/ds1_train.csv'
        eval_path = 'code/data/ds1_valid.csv'

        pred_path = '/Users/kimtaewan/Desktop/개인공부/ML_440/code/src/res/result.txt'

        main(train_path, eval_path, pred_path)