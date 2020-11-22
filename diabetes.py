import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

def decisionTree(X_train, X_test, y_train, y_test):
    model= DecisionTreeClassifier()
    model.fit(X_train, y_train)

    predict= model.predict(X_test)
    print('Accuracy using Decision tree classifier is : ', accuracy_score(y_test, predict))
    print('Confusion matrix using Decision tree classifier : ', confusion_matrix(y_test, predict))

    filename = 'diabetesDecisionTree'
    joblib.dump(model, filename)
    print('diabetesDecisionTree created!')
    return accuracy_score(y_test, predict)

def knn(X_train, X_test, y_train, y_test):
    model= KNeighborsClassifier()
    model.fit(X_train, y_train)

    predict= model.predict(X_test)
    print('Accuracy using K-Neighbors Classifier is : ', accuracy_score(y_test, predict))
    print('Confusion matrix using K-Neighbors Classifier : ', confusion_matrix(y_test, predict))

    filename = 'diabetesKnn'
    joblib.dump(model, filename)
    print('diabetesKnn created!')
    return accuracy_score(y_test, predict)

def RandomForest(X_train, X_test, y_train, y_test):
    model= RandomForestClassifier()
    model.fit(X_train, y_train)

    predict= model.predict(X_test)
    print('Accuracy using Random forest classifier is : ', accuracy_score(y_test, predict))
    print('Confusion matrix using Random forest classifier : ', confusion_matrix(y_test, predict))

    filename = 'diabetesRandomForest'
    joblib.dump(model, filename)
    print('diabetesRandomForest created!')
    return accuracy_score(y_test, predict)

def logistic(X_train, X_test, y_train, y_test):
    model= LogisticRegression()
    model.fit(X_train, y_train)

    predict= model.predict(X_test)
    print('Accuracy using Logistic regression is : ', accuracy_score(y_test, predict))
    print('Confusion matrix using logistic regression : ', confusion_matrix(y_test, predict))

    filename = 'diabetesLogistic'
    joblib.dump(model, filename)
    print('diabetesLogistic created!')
    return accuracy_score(y_test, predict)

def main():
    data = pd.read_csv('diabetes.csv')
    print(data.head())
    X= data.drop('Outcome', axis=1)
    y= data['Outcome']

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=1)

    logisticAccuracy= logistic(X_train, X_test, y_train, y_test)
    knnAccuracy= knn(X_train, X_test, y_train, y_test)
    RFAccuracy= RandomForest(X_train, X_test, y_train, y_test)
    DTreeAccuracy= decisionTree(X_train, X_test, y_train, y_test)

    algorithms= ['Logistic Regression', 'K-Nearest Neighbours', 'Random Forest', 'Decision Tree']
    accuracies=[logisticAccuracy, knnAccuracy, RFAccuracy, DTreeAccuracy]

    plt.bar(algorithms, accuracies)
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracies')
    plt.title('Algorithmwise accuracies for diabetes prediction')
    plt.show()

if __name__=='__main__':
    main()