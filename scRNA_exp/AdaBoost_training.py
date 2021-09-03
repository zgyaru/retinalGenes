import clean_data
import numpy as np

from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier

if __name__ == "__main__":
    test_number = 'test5'

    print(test_number)
    feature_data, positive_list, negative_list = clean_data.load_datasets(test_number)
    X, y = clean_data.construct_training_set(feature_data, positive_list, negative_list)
    kf = KFold(n_splits=5)

    # use Gaussian Naive Bayes
    bootstrap_list = [True, False]
    max_accuracy = -1000000
    final_estimators = ''
    final_bootstrap = 0

    va_accuracy_list = np.zeros(kf.get_n_splits()) 
    k = 0
    for train_index, validate_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", validate_index)
        X_train, X_validate = X[train_index], X[validate_index]
        y_train, y_validate = y[train_index], y[validate_index]

        model = AdaBoostClassifier()
        model.fit(X_train, y_train)
        va_accuracy_list[k] = model.score(X_validate, y_validate)
        k += 1
    print("accuarcy list is:", va_accuracy_list)
    accuracy = va_accuracy_list.mean()
    

    print("max accuracy is", accuracy)
