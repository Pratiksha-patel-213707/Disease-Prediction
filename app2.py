from flask import Flask, jsonify, render_template, request
import joblib
import jsonify
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

app = Flask(_name_)
model=pickle.load(open('random_forest_regression_model.pkl','rb'))

@app.route("/", methods=['GET'])
def index():
    return render_template("home.html")
standard_to =StandardScaler()
@app.route('/predict',methods=['POST'])
def result():

    murder= float(request.form['murder'])
    rape= float(request.form['rape'])
    kidnapping = float(request.form['kidnapping'])
    robbery= float(request.form['robbery'])
    station_location= float(request.form['station_location'])
    case_type= float(request.form['case_type'])

    X= np.array([[ murder,rape,kidnapping,robbery,station_location,
                  case_type ]])

    scaler_path= 'D:\\pbl\\random_forest_regression_model.pkl'

    sc=joblib.load(scaler_path)

    X_std= sc.transform(X)

    model_path='D:/pbl/Final - Copy.ipynb'

    model= joblib.load(model_path)

    Y_pred=model.predict(X_std)

    #return jsonify({'Prediction': float(Y_pred)})
    return render_template("predict.html", pred='Cases for the selected category will be {}'.format(Y_pred))

if _name_ == "_main_":
    app.run(debug=True)