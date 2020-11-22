import joblib
import numpy as np
print('Enter your details to find whether you are diabetes patient or not')
Pregnancies = int(input('Enter your Pregnancies : '))
if Pregnancies > 17 or Pregnancies < 0:
    print('wrong input')
    exit()

Glucose = int(input('Enter your Glucose : '))
if Glucose > 200 or Glucose < 0:
    print('wrong input')
    exit()

BloodPressure = int(input('Enter your BloodPressure : '))
if BloodPressure > 122 or BloodPressure < 0:
    print('wrong input')
    exit()

SkinThickness = int(input('Enter your SkinThickness : '))
if SkinThickness > 99 or SkinThickness < 0:
    print('wrong input')
    exit()

Insulin = int(input('Enter your Insulin : '))
if Insulin > 846 or Insulin < 0:
    print('wrong input')
    exit()

BMI = float(input('Enter your BMI : '))
if BMI > 67.1 or BMI < 0:
    print('wrong input')
    exit()

DiabetesPedigreeFunction = float(input('Enter your DiabetesPedigreeFunction : '))
if DiabetesPedigreeFunction > 2.42 or DiabetesPedigreeFunction < 0.078:
    print('wrong input')
    exit()

Age = int(input('Enter your Age : '))
if Age > 81 or Age < 21:
    print('wrong input')
    exit()

data= np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
       BMI, DiabetesPedigreeFunction, Age]])

model1= joblib.load('diabetesDecisionTree')
predict1= model1.predict(data)
if predict1 == [0]:
    print("You are not diabetes patient according to Decision Tree Classifier")
else:
    print("You are diabetes patient according to Decision Tree Classifier")


model2= joblib.load('diabetesKnn')
predict2= model2.predict(data)
if predict2 == [0]:
    print("You are not diabetes patient according to K-Nearest Neighbour Classifier")
else:
    print("You are diabetes patient according to K-Nearest Neighbour Classifier")


model3= joblib.load('diabetesLogistic')
predict3= model3.predict(data)
if predict3 == [0]:
    print("You are not diabetes patient according to Logistic regression")
else:
    print("You are diabetes patient according to Logistic regression")


model4= joblib.load('diabetesRandomForest')
predict4= model4.predict(data)
if predict4 == [0]:
    print("You are not diabetes patient according to Random Forest Classifier")
else:
    print("You are diabetes patient according to Random Forest Classifier")