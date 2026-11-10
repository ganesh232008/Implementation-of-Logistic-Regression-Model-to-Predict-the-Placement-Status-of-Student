# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:

1.Hardware – PCs

2.Anaconda – Python 3.7 Installation / Jupyter notebook

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: L Ganesh Kanna
RegisterNumber: 212225230072
*/

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv("Placement_Data.csv")

# View first 5 rows
print("First 5 rows of the dataset:")
print(data.head())
data1 = data.copy()

# Dropping 'sl_no' (serial number) and 'salary' (not needed for predicting placement)
data1 = data1.drop(["sl_no", "salary"], axis=1)

print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())
print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())

print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())
cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]

le = LabelEncoder()

for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])

print("\nData after Label Encoding:")
print(data1.head())
X = data1.iloc[:, :-1]
# y = 'status' column
y = data1["status"]

print("\nFeatures (X) sample:")
print(X.head())

print("\nTarget (y) sample:")
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)
lr = LogisticRegression(solver="liblinear")

# Train the model
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("\nPredicted values (y_pred):")
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Classification Report: precision, recall, F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]

new_prediction = lr.predict(new_student)

print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])
```

Output:
<img width="1208" height="315" alt="image" src="https://github.com/user-attachments/assets/b0beb01a-b55b-49a1-a818-a98277c6ab3a" />
<img width="1063" height="332" alt="image" src="https://github.com/user-attachments/assets/a993353f-5996-4e23-b2ef-04c7ffc235be" />
<img width="952" height="353" alt="image" src="https://github.com/user-attachments/assets/12f3a824-ee26-424e-b77d-6647d15dcfec" />
<img width="1090" height="396" alt="image" src="https://github.com/user-attachments/assets/36effcfe-2465-42dc-be08-f511ef9db7e5" />
<img width="1032" height="337" alt="image" src="https://github.com/user-attachments/assets/cfedf3c3-465f-4ba1-9e52-3989868324a1" />
<img width="1017" height="417" alt="image" src="https://github.com/user-attachments/assets/cbe57c54-2c57-4856-98ea-f1596befc216" />
<img width="1282" height="367" alt="image" src="https://github.com/user-attachments/assets/f6cdb638-9036-4e45-8401-d2f1e11b0670" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
