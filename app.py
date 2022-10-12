import pickle
from flask import Flask ,render_template,request,redirect,url_for,app,jsonify,flash,session,escape
from matplotlib import Scalar

import numpy
import numpy as np
import pandas

app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regmodel','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['Post'])

def predict_api():
    data=request.json['data']
    print (data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=Scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])
