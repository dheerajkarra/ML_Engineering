import os
import pickle
from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model

# from waitress import serve

def predict(features):    
    model = load_model('text_clf.h5', compile=False) 
    X = features
    preds = model.predict(X)    
    preds = np.argmax(preds)
        
    pred_dict = {0: 'Chardonnay',
     1: 'Pinot Noir',
     2: 'Cabernet Sauvignon',
     3: 'Red Blend',
     4: 'Bordeaux-style Red Blend',
     5: 'Sauvignon Blanc',
     6: 'Syrah',
     7: 'Riesling',
     8: 'Merlot',
     9: 'Zinfandel'}
    
    # return float(preds)
    return pred_dict[preds]

app = Flask('Text Classification')

@app.route('/predict', methods=['POST'])
def predict_endpoint():    
    features = request.get_json()
    print(features['key'])
    pred = predict(np.array([features['key']]))            
    
    result = {
        'text_clf_prediction': pred,
    }
    return jsonify(result)

if __name__ == "__main__":
    # app.run(debug=True, host='0.0.0.0', port=9696)
    app.run(debug=True, host='127.0.0.1', port=9696)
    # serve(app, listen='*:9696')