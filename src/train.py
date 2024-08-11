import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import feature_construct as fc

def train_RF(X_train,y_train):

    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10, 15],
        'max_features': ['sqrt', 'log2'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    randomized_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=44),
        param_distributions=param_grid,
        n_iter=100,  # Number of parameter settings sampled
        n_jobs=-1,
        scoring='accuracy',
        random_state=44,
        cv=5
        )

    randomized_search.fit(X_train_split, y_train_split)
    results = randomized_search.cv_results_

    results_df = pd.DataFrame({
        'mean_test_score': results['mean_test_score'],
        'std_test_score': results['std_test_score'],
        'params': results['params'],
        'rank_test_score': results['rank_test_score']
        })
    sorted_results_df = results_df.sort_values(by='rank_test_score')
    top_5_results = sorted_results_df.head(5)

    # Initialize an empty list to store the top models
    top_models = []
    result=[]
    # Loop through the top 5 results and build models using those parameters
    for idx, row in top_5_results.iterrows():
        params = row['params']
        model = RandomForestClassifier(**params)
        model.fit(X_train_split, y_train_split) 
        top_models.append(model)
        scores = cross_val_score(model, X_train_split, y_train_split, cv=5, scoring='accuracy')  # 5-fold cross-validation
        mean_accuracy = np.mean(scores)
        std_accuracy = np.std(scores)
        result.append((mean_accuracy,std_accuracy))
    print(sorted_results_df.head(5))
    return(top_models,result)

train_data = pd.read_csv('./input/train.csv')
test_data = pd.read_csv('./input/test.csv')
y_train = train_data['Survived']

X_Base_train=fc.benchmark(train_data)
X_Base_train=X_Base_train.drop(columns=['Survived'])
X_Base_test=fc.benchmark(test_data)
print("This is the result of benchmark model:")
benchmark_models,benchmark_result=train_RF(X_Base_train,y_train)


