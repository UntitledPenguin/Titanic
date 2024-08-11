import pandas as pd

def benchmark(X_input):
    #Features dropped in the benchmark model:
    X_input = X_input.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    #The missing value in Fare has been filled:
    X_input['Fare'].fillna(X_input['Fare'].mean(), inplace=True)
    #The missing value in Age in filled with mean:
    X_input['Age'].fillna(X_input['Age'].mean(), inplace=True)
    #Sex has been turned into one-hot format:
    X_input = pd.get_dummies(X_input, columns=['Sex'], drop_first=True)
    #The missing value in Embarked has been filled by the most frequent value:
    most_frequent_embarked = X_input['Embarked'].mode()[0]
    X_input['Embarked'].fillna(most_frequent_embarked, inplace=True)
    X_input = pd.get_dummies(X_input, columns=['Embarked'], drop_first=True)    

    return(X_input)

def featureconstruct(X_input):
    