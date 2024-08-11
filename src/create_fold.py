
import pandas as pd
from sklearn.model_selection import KFold

def get_fold(data,fold_number):
    n_splits = fold_number
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X_train_list = []
    X_val_list = []

    data['Fold']=-1

    for fold, (train_index, val_index) in enumerate(kf.split(data)):
        data.loc[val_index, 'Fold'] = fold

    for i in range(fold_number):
        X_train = data[data['Fold'] != i].drop(columns=['Fold'])
        X_val = data[data['Fold'] == i].drop(columns=['Fold', 'Survived'])
        X_train_list.append(X_train)
        X_val_list.append(X_val)
    return X_train_list, X_val_list
