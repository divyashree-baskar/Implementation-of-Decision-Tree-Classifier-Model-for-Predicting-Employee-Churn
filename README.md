# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the data from a CSV file.
2.Displays the first and last 5 rows.
3.Checks for missing values.
4.Shows how many employees left (1) vs stayed (0).
5.Encodes the categorical "salary" column (e.g., low = 0, medium = 1, high = 2).
6.x: independent variables (features)
7.y: dependent variable (target - whether the employee left)
8.Splits data into training (80%) and testing (20%).
9.Trains a Decision Tree using entropy (information gain) as the criterion.
10.Predicts on the test set.
11.Calculates the accuracy of the model
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: DIVYASHREE B
RegisterNumber:  212224040081
*/


import pandas as pd
data=pd.read_csv("Employee.csv")
print("Name: Divyashree B\nReg.no: 212224040081")
data.head()

data.tail()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years","salary"]]
x

y=data["left"]
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test, y_pred)
accuracy

dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
```

## Output:
<img width="1387" height="293" alt="image" src="https://github.com/user-attachments/assets/2a5cc493-d831-4795-8e1f-706ae7dd57b5" />


<img width="1346" height="309" alt="image" src="https://github.com/user-attachments/assets/83fe90dd-888b-45b7-8d15-48e21d6a5250" />

<img width="377" height="295" alt="image" src="https://github.com/user-attachments/assets/e1359b0a-6127-4804-9de3-8ac3050a7eff" />   <img width="332" height="77" alt="image" src="https://github.com/user-attachments/assets/0eb6788f-eeb5-471e-81bf-cb1275d0db88" />

<img width="1341" height="236" alt="image" src="https://github.com/user-attachments/assets/b99070ce-97e5-4f2f-9f8c-83b749dab5f1" />

<img width="1296" height="446" alt="image" src="https://github.com/user-attachments/assets/600ea73d-97ac-4dd4-b240-a7dad9e7fdc0" />

<img width="534" height="262" alt="image" src="https://github.com/user-attachments/assets/d251a8bd-bcfe-4aff-8ff2-db0b7c9d061e" />  <img width="282" height="29" alt="image" src="https://github.com/user-attachments/assets/42ad3380-c823-4ac2-92e9-dbf0f6554a1c" />

<img width="342" height="30" alt="image" src="https://github.com/user-attachments/assets/caed2474-eb45-43e8-b116-b4a135b1f87e" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
