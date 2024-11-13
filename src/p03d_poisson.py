import numpy as np
import util
import matplotlib.pyplot as plt
from linear_model import LinearModel

class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape

        def calc_grad(theta):
            # step_size = learning rate (LinearModel 모듈에 지정되어있음)
            return self.step_size * x.T @ (y-np.exp(x @ theta)) / m
        

        # Initialize theta
        if self.theta is None:
            theta = np.zeros(n)
        else:
            theta = self.theta

        iter_count = 0
        max_iter = 2000
        while iter_count < max_iter:
            prev_theta = theta.copy()  # 현재 theta 저장
            grad = calc_grad(theta)
        
            if np.linalg.norm(grad, 1) < self.eps:  # gradient가 충분히 작은지 확인
                break
            
            theta += grad
        
            # theta의 변화가 충분히 작은지 확인
            if prev_theta is not None:
                theta_diff = np.linalg.norm(theta - prev_theta, 1)
                if theta_diff < self.eps:
                    break

        self.theta = theta    
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***

def main():
    # Load training set
    x_train, y_train = util.load_dataset('code/data/ds4_train.csv', add_intercept=True)
    x_valid, y_valid = util.load_dataset('code/data/ds4_valid.csv', add_intercept=True)

    clf = PoissonRegression(step_size=2e-7)
    clf.fit(x_train,y_train)

    # 예측값 생성
    train_pred = clf.predict(x_train)
    valid_pred = clf.predict(x_valid)

    # 1. Training vs Validation Set 결과
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(y_train)), y_train, c='g', label='label', alpha=0.5)
    plt.scatter(range(len(train_pred)), train_pred, c='r', label='prediction', alpha=0.5)
    plt.title('Training Set')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(range(len(y_valid)), y_valid, c='g', label='label', alpha=0.5)
    plt.scatter(range(len(valid_pred)), valid_pred, c='r', label='prediction', alpha=0.5)
    plt.title('Validation Set')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig('code/src/res/Poisson_reg_train_valid_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Actual vs Predicted Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, train_pred, alpha=0.5)
    plt.plot([0, max(y_train)], [0, max(y_train)], 'r--')  # 45도 선
    plt.title('Actual vs Predicted (Training)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    plt.subplot(1, 2, 2)
    plt.scatter(y_valid, valid_pred, alpha=0.5)
    plt.plot([0, max(y_valid)], [0, max(y_valid)], 'r--')  # 45도 선
    plt.title('Actual vs Predicted (Validation)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    plt.tight_layout()
    plt.savefig('code/src/res/Poisson_reg_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()