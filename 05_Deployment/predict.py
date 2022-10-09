###################################################################
# Question 1
# pipenv, version 2022.10.4

###################################################################
# Question 2
# sha256:08ef968f6b72033c16c479c966bf37ccd49b06ea91b765e1cc27afefe723920b
# 08ef968f6b72033c16c479c966bf37ccd49b06ea91b765e1cc27afefe723920b

###################################################################
# Question 3

# import pickle

# def predict(features):

#     with open('dv.bin', 'rb') as f_in:
#         dv = pickle.load(f_in)
#     with open('model1.bin', 'rb') as f_in:
#         model = pickle.load(f_in)

#     X = dv.transform(features)
#     preds = model.predict_proba(X)[:, 1]
#     return float(preds)    

# test_data =  {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

# score = predict(test_data)
# print(score) # 0.16213414434326598

###################################################################
# Question 4

# import requests

# url = 'http://127.0.0.1:9696/predict'
# # url = 'http://localhost:9696/predict'

# client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
# response = requests.post(url, json=client)
# print(response.json())
# # {'cc_prediction': 0.9282218018527452}


###################################################################
# Question 5

# REPOSITORY                 TAG                IMAGE ID       CREATED       SIZE
# svizor/zoomcamp-model      3.9.12-slim        571a6fdc554b   6 days ago    125MB

###################################################################
# Question 6

import requests

url = 'http://127.0.0.1:9696/predict'
# url = 'http://localhost:9696/predict'

client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
response = requests.post(url, json=client)
print(response.json())

# {'cc_prediction': 0.7692649226628628}