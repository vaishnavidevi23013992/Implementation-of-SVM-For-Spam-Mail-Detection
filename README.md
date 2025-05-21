# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.

2.Import the dataset to operate on.

3.Split the dataset.

4.Predict the required output.

5.End the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VAISHNAVIDEVI V
RegisterNumber:212223040230
*/
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
```

## Output:
#### Result Output:
![image](https://github.com/POZHILANVD/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870498/ebca3817-9ad2-4374-bf00-f29b9b6d0598)
#### data.head():
![image](https://github.com/POZHILANVD/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870498/482bcc29-05cd-4eaf-bd7a-a0ae5f96deb9)
#### data.info():
![image](https://github.com/POZHILANVD/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870498/78b3b8ad-750e-424a-853f-b876add8be72)
#### data.isnull().sum():
![image](https://github.com/POZHILANVD/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870498/bf878d0f-b9cf-429d-a9bd-a6c64aeb9357)
![image](https://github.com/POZHILANVD/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870498/ca376c9b-f59d-46eb-b97a-e751e5a05d38)
#### Y_prediction Value:
![image](https://github.com/POZHILANVD/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870498/b1b11a82-da48-4063-85cb-ee7bfef2860c)
#### Accuracy Value:
![image](https://github.com/POZHILANVD/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870498/c090233c-85d3-4285-a19a-f6b8e4327d60)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
