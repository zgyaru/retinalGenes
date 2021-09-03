import clean_data
import numpy as np

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    test_number = 'test5'

    print(test_number)
    feature_data, positive_list, negative_list = clean_data.load_datasets(test_number)
    X, y = clean_data.construct_training_set(feature_data, positive_list, negative_list)
    kf = KFold(n_splits=5)

    # use random forest
    bootstrap_list = [True, False]
    max_accuracy = -1000000
    final_estimators = ''
    final_bootstrap = 0

    for estimators in range(5, 101, 5):
        for bootstrap in bootstrap_list: 
            va_accuracy_list = np.zeros(kf.get_n_splits()) 
            k = 0
            for train_index, validate_index in kf.split(X):
            # print("TRAIN:", train_index, "TEST:", validate_index)
                X_train, X_validate = X[train_index], X[validate_index]
                y_train, y_validate = y[train_index], y[validate_index]

                model = RandomForestClassifier(n_estimators=estimators, bootstrap=bootstrap)
                model.fit(X_train, y_train)
                va_accuracy_list[k] = model.score(X_validate, y_validate)
                k += 1
            accuracy = va_accuracy_list.mean()
            # print('estimators = ', estimators,', bootstrap: ',bootstrap, 'and accuracy is ', accuracy)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                final_estimators = estimators
                final_bootstrap = bootstrap
    

    print("max accuracy is", max_accuracy)
    print("number of trees in the forest", final_estimators)
    print("bootstrap is", final_bootstrap)
