# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the libraries and read the data frame using pandas.
   
2. Calculate the null values present in the dataset and apply label encoder.
   
3. Determine test and training data set and apply decison tree regression in dataset.
   
4. Calculate Mean square error,data prediction and r2.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: S Kantha Sishanth
RegisterNumber: 212222100020 
```
```py
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
l0=LabelEncoder()

data["Position"]=l0.fit_transform(data['Position'])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:

### data.head()

![ml_7 1](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118298456/99cbe989-12a0-4ce2-affa-7808c7494fce)


### data.info()

![ml_7 2](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118298456/cad3230e-a3a5-49c1-9bf3-e77b197613e9)


### isnull() and sum()

![ml_7 3](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118298456/fc7877d9-85f2-4975-b4b5-9e67b71b6313)


### data.head() for salary

![ml_7 4](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118298456/8303341c-b203-45ae-b557-da107c0afd07)


### MSE Value

![ml_7 5](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118298456/68fc07ff-206f-428b-b970-0bba739501b8)


### r2 value

![ml_7 6](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118298456/56acc727-4989-425e-89f1-c4262a7bbbd4)


### data prediction

![ml_7 7](https://github.com/Skanthasishanth/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118298456/69663cad-09f9-4bfd-95b5-73ec72fdd5dd)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
