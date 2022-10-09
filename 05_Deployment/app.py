import os
import pickle
from flask import Flask, request, jsonify
# from waitress import serve

def predict(features):

    with open('dv.bin', 'rb') as f_in:
        dv = pickle.load(f_in)
    with open('model2.bin', 'rb') as f_in:
        model = pickle.load(f_in)

    X = dv.transform(features)
    preds = model.predict_proba(X)[:, 1]
    return float(preds)    


app = Flask('credit_card_prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():    
    features = request.get_json()
    # features = prepare_features(energy_usage)
    pred = predict(features)
    
    result = {
        'cc_prediction': pred,
    }
    return jsonify(result)

if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0', port=9696)
    app.run(debug=True, host='127.0.0.1', port=9696)
    # serve(app, listen='*:9696')