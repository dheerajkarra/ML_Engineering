import os
import pickle
from flask import Flask, request, jsonify
# from waitress import serve

def predict(features):
    with open('model_files.bin', 'rb') as f_in:
        (model,scaler) = pickle.load(f_in)
 
    X = features
    preds = model.predict_proba(X)[:, 1]
    # preds = model.predict(X)
    return float(preds)

app = Flask('cc_fraud_detection')

import numpy as np

@app.route('/predict', methods=['POST'])
def predict_endpoint():    
    features = request.get_json()
    print(features['key'])
    pred = predict(np.array([features['key']]))
    # pred = predict(np.array(features['key']))
    
    result = {
        'fraud_prediction': pred,
    }
    return jsonify(result)

if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0', port=9696)
    app.run(debug=True, host='127.0.0.1', port=9696)
    # serve(app, listen='*:9696')