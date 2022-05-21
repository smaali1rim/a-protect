import pickle

import joblib
from flask import Flask, request, render_template, jsonify
import json
import numpy as np

model = joblib.load(r'C:\Users\user/boosting1_model')
app = Flask(__name__)

@app.route('/')
def index():
   return 'hello'
@app.route('/predict',methods=['POST'])
def predict():
    permissions = json.loads(request.data)
    values = permissions["permissions"]
    values = list(map(np.float, values))
    pre = np.array(values)
    pre = pre.reshape(1, -1)
    res = model.predict(pre)

    return jsonify({'prediction': str(res[0])})



if __name__ == '__main__':
    app.run(debug=True)