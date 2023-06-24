from flask import Flask,redirect,url_for,render_template,request
import numpy as np
import pickle
import sklearn
import os

file_path = '1.pkl'

if os.path.isfile(file_path):
    print("File exists")
else:
    print("File does not exist")

model = pickle.load(open('amit.pkl', 'rb'))

# Call the 'predict' method on the model
# prediction = model.predict(input_data)

app = Flask(__name__)   

@app.route('/')
def welcome():
    return render_template('home.html')


@app.route('/predict',methods = ["POST"])
def home():
    sy1 = request.form['s1']
    sy2 = request.form['s2']
    sy3 = request.form['s3']
    sy4 = request.form['s4']
    sy5 = request.form['s5']
    sy6 = request.form['s6']
    sy7 = request.form['s7']
    sy8 = request.form['s8']
    sy9 = request.form['s9']
    sy10 = request.form['s10']


    arr = np.array([[sy1,sy2,sy3,sy4,sy5,sy6,sy7, sy8, sy9,sy10]])
    # pred = file_path.predict(arr)
    pred = model.predict(arr)
    return render_template('after.html',data=pred)

if __name__ == '__main__':
    app.run(debug=True)