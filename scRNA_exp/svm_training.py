import clean_data
import numpy as np

from sklearn.model_selection import KFold
from sklearn import svm

if __name__ == "__main__":
    test_number = 'test1'

    print(test_number)
    feature_data, positive_list, negative_list = clean_data.load_datasets(test_number)
    X, y = clean_data.construct_training_set(feature_data, positive_list, negative_list)
    n , d = X.shape
    kf = KFold(n_splits=5)

    # use SVM
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    min_error = 1000000
    min_kernel = ''
    min_degree = 0

    for kernel in kernels:
        for degree in range(2, 7): 
            va_error_list = np.zeros(kf.get_n_splits()) 
            k = 0
            for train_index, validate_index in kf.split(X):
            # print("TRAIN:", train_index, "TEST:", validate_index)
                X_train, X_validate = X[train_index], X[validate_index]
                y_train, y_validate = y[train_index], y[validate_index]

                model = svm.SVC(kernel=kernel, degree=degree)
                model.fit(X_train, y_train)
                yhat = model.predict(X_validate)
                unequal_num = np.sum(yhat != y_validate)
                va_error_list[k] = unequal_num / y_validate.size
                k += 1
            error = va_error_list.mean()
            if error < min_error:
                min_error = error
                min_kernel = kernel
                min_degree = degree
    

    print("min error is ", min_error)
    print("min kernel is ", min_kernel)
    print("min degree is ", min_degree)
