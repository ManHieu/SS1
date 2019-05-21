from os.path import split
import numpy

from read_data import read_data

raw_data = read_data('x28.txt')
# print(raw_data)

data = []

for item in raw_data:
    line = item.split()
    # row = []

    # for i in line:
    #     row.append(float(i))

    data.append(line)

X = []
Y = []
for item in data:
    row_x = item[1:-1]
    X.append(row_x)
    # print(row_x)
# print(data)
# t = numpy.array(X)
# print(t)


def normalize_and_add_one(data):
    X = numpy.array(data)
    X_max = numpy.array([[numpy.array(X[:, column_id])
                          for column_id in range(X.shape[1])]
                         for _ in range(X.shape[0])])

    X_min = numpy.array([[numpy.amin(X[:, column_id])
                          for column_id in range(X.shape[1])]
                         for _ in range(X.shape[0])])

    x_normalized = (X-X_min)/(X_max-X_min)

    ones = numpy.array([[1] for _ in range(x_normalized.shape[0])])

    return numpy.column_stack((ones, x_normalized))


class RidgeRegression(object):
    def __init__(self):
        return

    def fit(self, X_train, Y_train, LAMBDA):
        assert len(X_train.shape) == 2 and X_train.shape[0] == Y_train.shape[0]

        W = numpy.linalg.inv(
            X_train.transpose().dot(X_train) +
            LAMBDA * numpy.identity(X_train.shape[1])
        ).dot(X_train.transpose()).dot(Y_train)

        return W

    def predict(self, W, X_new):
        X_new = numpy.array(X_new)
        Y_new = X_new.dot(W)
        return Y_new

    def compute_RSS(self, Y_new, Y_predicted):
        loss = 1. / Y_new.shape[0] * \
            numpy.sum((Y_new - Y_predicted)**2)

        return loss

    def get_the_best_LAMBDA(self, X_train, Y_train):
        def cross_validation(num_folds, LAMBDA):
            row_ids = numpy.array(range(X_train.shape[0]))
            vaild_ids = numpy.split(row_ids[:len(row_ids) - len(row_ids) % num_folds], num_folds)
            vaild_ids[-1] = numpy.append(vaild_ids[-1], row_ids[len(row_ids) - len(row_ids) % num_folds:])
            train_ids = [[k for k in row_ids if k not in vaild_ids[i]] for i in range(num_folds)]
            aver_RSS = 0

            for i in range(num_folds):
                valid_part = {'X': X_train[vaild_ids[i]], 'Y': Y_train[vaild_ids[i]]}
                train_part = {'X': X_train[train_ids[i]], 'Y': Y_train[train_ids[i]]}
                W = self.fit(train_part['X'], train_part['Y'], LAMBDA)
                Y_predicted = self.predict(W, valid_part['X'])
                aver_RSS += self.compute_RSS(valid_part['Y'], Y_predicted)
            
            return aver_RSS/num_folds

        def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_value):
            for current_LAMBDA in LAMBDA_value:
                aver_RSS = cross_validation(num_folds=5, LAMBDA=current_LAMBDA)
                if aver_RSS < minimum_RSS:
                    best_LAMBDA = current_LAMBDA
                    minimum_RSS = aver_RSS
                
            return best_LAMBDA, minimum_RSS
            

        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=0, minimum_RSS=1000**2,
                                              LAMBDA_value=range(50))
        
        LAMBDA_value = [k*1. / 1000 for k in range(
            max(0, (best_LAMBDA-1)*1000), (best_LAMBDA+1)*1000, 1
        )]
    
        best_LAMBDA, minimum_RSS = range_scan(best_LAMBDA=best_LAMBDA, minimum_RSS=minimum_RSS,
                                              LAMBDA_value=LAMBDA_value)
        
        return best_LAMBDA


if __name__ == '__main__':
    X = normalize_and_add_one(X)
    X_train, Y_train = X[:50], Y[:50]
    X_test, Y_test = X[50:], Y[50:]

    ridge_regression = RidgeRegression()
    best_LAMBDA = ridge_regression.get_the_best_LAMBDA(X_train, Y_train)
    print(best_LAMBDA)

    W_learned = ridge_regression.fit(
        X_train, Y_train, best_LAMBDA
    )

    Y_predicted = ridge_regression.predict(W_learned, X_test)
