import requests

url = 'http://127.0.0.1:9696/predict'
# url = 'http://localhost:9696/predict'

test_data = [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,  848,  597,  304,   90,  790,  402, 2210,  898,  644,  122,
            1,   11,   36,   55,  184,  151,    4,   66,   54,  118,   53,
          324,  259,  626,    2,  147,  379,   18]

# test_data = [
#             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#             0,    0,    0,   26,   76,   42,   28,   40,  996,  554, 1777,
#           72,    3,  106,   38, 1879, 1214,   20,  189,   71, 1201,  112,
#           12,   62,  893,   28, 1385,  915,  713]

json_data = {}
json_data['key'] = test_data

response = requests.post(url, json=json_data)
# response = requests.post(url, data=json.dumps(json_data))
print(response.json())

# Final Run

# test_data_x = np.array(
# [[   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,  848,  597,  304,   90,  790,  402, 2210,  898,  644,  122,
#            1,   11,   36,   55,  184,  151,    4,   66,   54,  118,   53,
#          324,  259,  626,    2,  147,  379,   18],
#        [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
#            0,    0,    0,   26,   76,   42,   28,   40,  996,  554, 1777,
#           72,    3,  106,   38, 1879, 1214,   20,  189,   71, 1201,  112,
#           12,   62,  893,   28, 1385,  915,  713]])

# test_data_y = 
# array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]], dtype=float32)