import clean_data
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier


from sklearn import svm

if __name__ == "__main__":
    test_number = 'test5'

    print(test_number)
    feature_data, positive_list, negative_list = clean_data.load_datasets(test_number)
    X, y = clean_data.construct_training_set(feature_data, positive_list, negative_list)
    n , d = X.shape
    kf = KFold(n_splits=5)

    # use NN
    hyper_p_list = [(100,),(200,),(300,),(400,),(500,),(600,),(700,),(800,)]
    min_error = 1000000
    min_hyper = (1,)

    for hyper_p in hyper_p_list: 
        va_error_list = np.zeros(kf.get_n_splits()) 
        k = 0
        for train_index, validate_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", validate_index)
            X_train, X_validate = X[train_index].astype(np.float64), X[validate_index].astype(np.float64)
            y_train, y_validate = y[train_index].astype(np.float64), y[validate_index].astype(np.float64)

            model = MLPClassifier(hidden_layer_sizes=hyper_p , max_iter = 1000)
            model.fit(X_train, y_train)
            yhat = model.predict(X_validate)
            unequal_num = np.sum(yhat != y_validate)
            # print('unequal number is', unequal_num)
            va_error_list[k] = unequal_num / y_validate.size
            k += 1
        error = va_error_list.mean()
        print('parameter = ', hyper_p, 'and error is ', error)
        if error < min_error:
            min_error = error
            min_hyper = hyper_p


    print("min error is ", min_error)
    print("min hyper is ", min_hyper)

