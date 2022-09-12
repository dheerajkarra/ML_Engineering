import os
import pandas as pd
import numpy as np


os.chdir('F:\\Projects\\MLE\\01_Intro\\')
# conda create -n mle python=3.9

cars = pd.read_csv('data.csv')

# Question 1:
np.__version__
# '1.22.4'


# Question 2:
cars.shape[0]
# 11914

# Question 3:
cars['Make'].unique()
cars_q3 = cars.groupby(['Make'],as_index=False).count()
print(cars_q3.nlargest(3,'Model'))

#           Make  Model  Year  ...  city mpg  Popularity  MSRP
# 9    Chevrolet   1123  1123  ...      1123        1123  1123
# 14        Ford    881   881  ...       881         881   881
# 46  Volkswagen    809   809  ...       809         809   809

# Question 4:
cars_q4 = cars[cars['Make']=='Audi']
cars_q4['Make'].unique() # Audi
cars_q4['Model'].nunique() # 34

# Question 5:
cars.info()
cars.describe()
cars.isnull().any().sum() # 5

# Question 6:
cars['Engine Cylinders'].isnull().sum() # 30    
cars['Engine Cylinders'].median() # 6.0
cars['Engine Cylinders'].value_counts() # 4.0 
cars['Engine Cylinders'].mode() # 4.0 
mfv = cars['Engine Cylinders'].value_counts().index[0]
cars['Engine Cylinders'].fillna(mfv, inplace=True)
cars['Engine Cylinders'].isnull().sum() # 0    
cars['Engine Cylinders'].median() # 6.0

# Median not changed after replacing missing values - No


# This is related to applying Linear Regression using its analytical solution
# which is done via Normal equation -  Inverse(XT X).(XT y)

# Question 7:
cars_7 = cars[cars['Make'] == 'Lotus']
cars_7 = cars_7[["Engine HP", "Engine Cylinders"]]
cars_7 = cars_7.drop_duplicates() # 9 rows
X = np.array(cars_7)
XTX = np.dot(X.T,X)
# det = np.linalg.det(XTX) # 4556268.000000008
XTX_inv = np.linalg.inv(XTX)
XTX_result = np.dot(XTX_inv,X.T)

y = [1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800]
w = np.dot(XTX_result,y) # array([  4.59494481, -63.56432501])













