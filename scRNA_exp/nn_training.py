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
    hyper_p_list = [(5, 2), (5, 3), (6, 2), (6, 3), (7, 2), (15, ), (100,),(200,),(300,),(400,),(500,),(600,),(700,),(800,)]
    # hyper_p_list = []

    max_accuracy = -1000000
    final_hyper = (1,)

    for hyper_p in hyper_p_list: 
        va_accuracy_list = np.zeros(kf.get_n_splits()) 
        k = 0
        for train_index, validate_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", validate_index)
            X_train, X_validate = X[train_index].astype(np.float64), X[validate_index].astype(np.float64)
            y_train, y_validate = y[train_index].astype(np.float64), y[validate_index].astype(np.float64)

            model = MLPClassifier(hidden_layer_sizes=hyper_p , max_iter = 10000)
            model.fit(X_train, y_train)
            va_accuracy_list[k] = model.score(X_validate, y_validate)
            k += 1
        accuracy = va_accuracy_list.mean()
        # print('parameter = ', hyper_p, 'and accuracy is ', accuracy)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            final_hyper = hyper_p


    print("max accuracy is", max_accuracy)
    print("hyper is", final_hyper)

