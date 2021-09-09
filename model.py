import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import linear_model, tree, ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC

dataframe=pd.read_csv("./heart.csv")
categorical_val = []
continous_val = []
for column in dataframe.columns:

    if len(dataframe[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)

df = dataframe.dropna()
df = df.drop(columns = ['oldpeak', 'slope', 'ca', 'thal', 'fbs', 'restecg', 'exang'])
df = df.rename(columns = {'age': 'age', 'sex': 'gender', 'cp': 'chest pain', 'trestbps': 'blood pressure', 'chol': 'cholestrol level', 'thalach': 'max heart rate', })

dataset = pd.get_dummies(dataframe, columns = ['sex', 'cp', 'fbs', 'ca'])
s_sc = StandardScaler()
col_to_scale = ['age', 'chol', 'thalach']
dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])

X= df.drop(['target'], axis=1)
y= df['target']
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.05,random_state=40)

svm=SVC(C=56,kernel='linear')
model4=svm.fit(X_train,y_train)
prediction4=model4.predict(X_test)

import pickle
pickle.dump(model4, open('heart_disease_detector.pkl', 'wb'))# load model
heart_disease_detector_model = pickle.load(open('heart_disease_detector.pkl', 'rb'))

from flask import Flask, request, render_template
model = pickle.load(open('heart_disease_detector.pkl', 'rb')) 
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('./home.html')

@app.route('/predict', methods =['POST'])
def predict():

    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model.predict(array_features)
    
    output = prediction
    
    if output == 1:
        return render_template('Heart Disease Classifier.html', 
                               result = 'The patient is not likely to have heart disease!')
    else:
        return render_template('Heart Disease Classifier.html', 
                               result = 'The patient is likely to have heart disease!')

if __name__ == '__main__':
#Run the application
    app.run()

    
    