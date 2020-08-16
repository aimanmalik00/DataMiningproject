import sys
from flask import Flask
from flask import request, redirect , render_template
import pandas as pd
# from numpy import *
import numpy as np
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt


app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World from Flask!"

@app.route('/test', methods = ['POST'])
def test():
    df = pd.read_csv('./data/test dataset.csv')
    


    #quality prediction for ggod and bad
    # df['quality'] = df['quality'].map({3 : 'bad', 4 :'bad', 5: 'bad',6: 'good', 7: 'good', 8: 'good'})
    # df['quality'].value_counts()

    # quality prediction from 1 to 10
    # df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]

    # X = df.drop(['quality','goodquality'], axis = 1)
    # y = df['goodquality']

    # df['goodquality'].value_counts()

    #model
    #logisticregression
    x = df.iloc[:,:11]
    y = df.iloc[:,11]

    # determining the shape of x and y.
    # print(x.shape)
    # print(y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    #model train
    model = LogisticRegression()

    model.fit(x_train, y_train)

    # predicting the results for the test set
    y_pred = model.predict(x_test)

    # calculating accuracy
    # print("Training accuracy :", model.score(x_train, y_train))
    # print("Testing accuracy :", model.score(x_test, y_test))
    predict = []
    facidity = request.form['facidity']
    vacidity = request.form['vacidity']
    citricacid = request.form['citricacid']
    sugar = request.form['sugar']
    chlorides = request.form['chlorides']
    fsd = request.form['fsd']
    tsd = request.form['tsd']
    density = request.form['density']
    ph = request.form['ph']
    sulphates = request.form['sulphates']
    alcohol = request.form['alcohol']
   


    
    
    predict.append(facidity)
    predict.append(vacidity)
    predict.append(citricacid)
    predict.append(sugar)
    predict.append(chlorides)
    predict.append(fsd)
    predict.append(tsd)
    predict.append(density)
    predict.append(ph)
    predict.append(sulphates)
    predict.append(alcohol)

    # prediction = prediction_method(predict)
    

    predict_array =[]
    for i in range(0, len(predict)): 
        predict_array.append(float(predict[i]))

    predict_array = np.array(predict_array, ndmin=2)
    predict_result = model.predict(predict_array)
    if(predict_result > 5):
        predict_result = "Good"
    else:
        predict_result = "Bad"
    return render_template("result.html" , result =predict_result  ) 


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000 , debug=True)


