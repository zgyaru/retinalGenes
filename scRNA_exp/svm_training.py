import clean_data
import numpy as np

from sklearn.model_selection import KFold
from sklearn import svm

if __name__ == "__main__":
    test_number = 'test5'

    print(test_number)
    feature_data, positive_list, negative_list = clean_data.load_datasets(test_number)
    X, y = clean_data.construct_training_set(feature_data, positive_list, negative_list)
    n , d = X.shape
    kf = KFold(n_splits=5)

    # use SVM
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    max_accuracy = -1000000
    final_kernel = ''
    final_degree = 0

    for kernel in kernels:
        for degree in range(2, 7): 
            va_accuracy_list = np.zeros(kf.get_n_splits()) 
            k = 0
            for train_index, validate_index in kf.split(X):
            # print("TRAIN:", train_index, "TEST:", validate_index)
                X_train, X_validate = X[train_index], X[validate_index]
                y_train, y_validate = y[train_index], y[validate_index]

                model = svm.SVC(kernel=kernel, degree=degree)
                model.fit(X_train, y_train)
                va_accuracy_list[k] = model.score(X_validate, y_validate)
                k += 1
            accuracy = va_accuracy_list.mean()
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                final_kernel = kernel
                final_degree = degree
    

    print("max accuracy is", max_accuracy)
    print("final kernel is", final_kernel)
    print("final degree is", final_degree)
