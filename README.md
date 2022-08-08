
# Heart Disease Prediction

This project uses the Machine Learning algorithm
to predict whether a person will get a loan passed 
in the bank or not, where the dataset has all the
input parameters required for prediction.

##  About The Dataset

The dataset comprises of the different values under 
which the bank decides whether a person is eligible
to get a loan passed or not.
Some the sample values are:
1. Applicant Income
2. Dependents
3. Property owned
4. Loan Term etc....

Basically this dataset is the collection of the raw values
which were collected previously for all types of 
customers and then were recorded to see if loan was granted or not.

## Data Analysis Details

### 1. Null Values:
In the project we havae firstly handled all the Null 
values and replaced them with the most frequently occuring
values of their subsequent rows.

### 2. Training the Algorithm:
In order to find the best possible accuray we have trained
or dataset in four different Machine Learning Algorithms
and then finally trained the input values with the 
model to predict the perfect answer in Yes or No

### 3. Logistic Regression:
After standard analysis finally the Logistic Regression classifier has given the highest test set accuracy. So, I have decided to move ahead with this model.
The final prediction will be in the form of 0(NOT GRANTED) and 1(GRANTED).

## Project Explanation

This project can basically be used as tool for 
the end users or the bank employees to predict 
whether the person will be able to pass the Loan criterion
and this can be a great time saving hack for the 
upcoming increase in Digitalization as well as population.

To make my project more worthy I have used the concept of 
input data which can be directly used by the user in which, he 
just has to enter the required values and Bingo!! the model will
definetely predict if you are eligible for the grant within seconds.

## Deployment

To deploy this project run

```bash
### For better understanding use .ipynb format given above

#Loan Status Predicting

###Importing the dataset and Libraries

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('/content/Loan_status.csv')
df.drop(columns=['Loan_ID'], inplace=True)
df.shape

df.head(2)

##Data Preprocessing

###1. Replacing the Values and adding the columns

df.replace(to_replace='3+', value=4, inplace=True)

df['Total_Income'] = df.insert(5, "Total_Income", None)
df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df.drop(columns=['ApplicantIncome', 'CoapplicantIncome'], inplace=True)
df.head(2)

###2. Replacing the null values with mode.

df.isnull().sum()

null_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']

for columns in null_cols:
  df[columns]= df[columns].fillna(df[columns].mode()[0])

df.isnull().sum()

###3. Encoding the Categorical Variables with the help of Label Encoder

cat_var = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for columns in cat_var:
  df[columns] = le.fit_transform(df[columns])

df.tail()

##Training our data for machine Learning Models.

###1. Defining the dpenden and Independent variables

x = df.drop(columns=['Loan_Status'], axis=1).values
y = df['Loan_Status'].values

print(x.shape)
print(y.shape)

###2. Spliting into Training and test sets.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

##3. Defining in a Standardized method to fit all the Machine Learning Models

def classify(classifier, x, y):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
  classifier.fit(x_train, y_train)
  print(classifier.predict(x_test))
  print('Accuracy Score Training Set: ', accuracy_score(classifier.predict(x_train).round(), y_train))
  print('Accuracy Score Test Set: ', accuracy_score(classifier.predict(x_test).round(), y_test))


###4. (i) Linear Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classify(classifier, x, y)

###5. (ii) Decision tree classification

from sklearn.tree import DecisionTreeClassifier
classifier  = DecisionTreeClassifier()
classify(classifier, x, y)

###6. (iii) Random Forest Classification.

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classify(classifier, x, y)

#### Since we got the best Accuracy Score in Linear Regression, so we will move ahead with Linear Regression model

df.tail(2)

## Testing our Model with input given as a new datapoints.

# Example 1: for the condition on Loan Granted

input_data_raw = (1,1,0,0,0,6200,120,180,1,2)
input_data = np.asarray(input_data_raw).reshape(1,-1)
classifier_1 = LogisticRegression()
classifier_1.fit(x_train, y_train)
prediction = classifier_1.predict(input_data)
print(prediction)

if (prediction == 0):
  print('No Loan Granted')
else:
  print('Loan Granted')

## Example 2: for the condition of loan not granted
# (same dataset but Loan amount increased a lot)

input_data_raw = (1,1,0,0,0,6200,12000,180,1,2)
input_data = np.asarray(input_data_raw).reshape(1,-1)
classifier_2 = LogisticRegression()
classifier_2.fit(x_train, y_train)
prediction = classifier_2.predict(input_data)
print(prediction)

if (prediction == 0):
  print('No Loan Granted')
else:
  print('Loan Granted')


```


## Conclusion

Now, since our model is trained perfectly 
therefore, it can be great helping hand for the banking staff
to predict if the customer at their door is fit for a loan grant or 
not, and would surely help the person as well to save his time and energy.

# Hi, I'm Avichal Srivastava ! 👋

You can reach out to me at: srivastavaavichal007@gmail.com 
LinkedIn: www.linkedin.com/in/avichal-srivastava-186865187

## 🚀 About Me
I'm a Mechanical Engineer by education, and I love to work with data, eventually started my coding journey for one of my Drone project, where I realized that it is something which makes me feel happy in doing, then I planed ahead to move in the Buiness Analyst or Data Analyst domain. The reason for choosing these domains is because I love maths a lot and all the Machine Learning algorithms are completely based on mathematical intution, So this was about me Hope! You liked it, and it is just a beginning, many more to come, till then Happy Analysing!!

