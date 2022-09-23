import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('F:\\Projects\\ML_Engineering\\02_Regression\\')

# California Housing Prices data
df = pd.read_csv('housing.csv')

sel_cols = ['latitude',
'longitude',
'housing_median_age',
'total_rooms',
'total_bedrooms',
'population',
'households',
'median_income',
'ocean_proximity',
'median_house_value']

df = df[sel_cols]

df = df.fillna(0)
df.isnull().sum()   

df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

###################################################################
# Question 1
df['ocean_proximity'].value_counts()
df['ocean_proximity'].mode() # <1H OCEAN

###################################################################
# Question 2
df.dtypes

from sklearn.model_selection import train_test_split

data_class = df.copy()
mean = data_class['median_house_value'].mean()

data_class['above_average'] = np.where(data_class['median_house_value']>=mean,1,0)
data_class = data_class.drop('median_house_value', axis=1)

df_train_full, df_test = train_test_split(data_class, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Correlation
df_numeric = df_train.copy()
df_numeric.dtypes
df_numeric = df_numeric.drop(["ocean_proximity","above_average"], axis=1)
df_numeric.describe()
df_numeric.corr()

plt.figure(figsize=(15,10))  
sns.heatmap(df_numeric.corr(),annot=True,linewidths=.5, cmap="Blues")
plt.title('Heatmap showing correlations among variables of numerical df')
plt.show()

df_corr = df_numeric.corr().unstack().sort_values(ascending = False)

# total_bedrooms            households                  0.979399
# households                total_bedrooms              0.979399
# total_bedrooms            total_rooms                 0.931546
# total_rooms               total_bedrooms              0.931546
# households                total_rooms                 0.921441
# total_rooms               households                  0.921441
# households                population                  0.906841
# population                households                  0.906841

df_corr[:50]
df_corr[df_corr.index == (           'population_per_household',           'total_rooms')]
# -0.029452

# total_bedrooms and households

###################################################################
# Question 3

from sklearn.metrics import mutual_info_score
cat = ['ocean_proximity']
def calculate_mi(series):
    return mutual_info_score(series, df_train.above_average)

df_mi = df_train[cat].apply(calculate_mi)
df_mi = df_mi.sort_values(ascending=False).to_frame(name='MI')
df_mi
#                        MI
# ocean_proximity  0.101384
round(df_mi,2) # 0.1

###################################################################
# Question 4

y_train = df_train.above_average.values
y_val = df_val.above_average.values
y_test = df_test.above_average.values

df_train = df_train.drop('above_average', axis=1)
df_val = df_val.drop('above_average', axis=1)
df_test = df_test.drop('above_average', axis=1)


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

num = sel_cols.copy()
num.remove('ocean_proximity')
num.remove('median_house_value')

train_dict = df_train[cat + num].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)
model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

val_dict = df_val[cat + num].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict(X_val)

accuracy = np.round(accuracy_score(y_val, y_pred),2)
print(accuracy) # 0.84

###################################################################
# Question 5

features = cat + num
features

orig_score = accuracy

df_elim = pd.DataFrame()
for c in features:
    subset = features.copy()
    subset.remove(c)
    
    train_dict = df_train[subset].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    dv.fit(train_dict)

    X_train = dv.transform(train_dict)

    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    val_dict = df_val[subset].to_dict(orient='records')
    X_val = dv.transform(val_dict)

    y_pred = model.predict(X_val)

    score = accuracy_score(y_val, y_pred)
    
    df_elim_temp = pd.DataFrame([c, (orig_score - score), score])    
    df_elim_temp = df_elim_temp.T
    df_elim_temp.columns = ['feature','score_reduced','score']
    df_elim = pd.concat([df_elim,df_elim_temp])
        
    print(c, orig_score - score, score)

df_elim = df_elim.sort_values(['score_reduced'])

#               feature score_reduced     score
# 0         total_rooms      0.003275  0.836725
# 0      total_bedrooms      0.003517  0.836483
# 0            latitude       0.00594   0.83406
# 0           longitude      0.006182  0.833818
# 0          households      0.008605  0.831395
# 0  housing_median_age      0.009816  0.830184
# 0          population      0.019748  0.820252
# 0     ocean_proximity      0.022655  0.817345
# 0       median_income      0.056812  0.783188

# total_rooms

###################################################################
# Question 6

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

data = df.copy()
data['median_house_value']=np.log1p(data['median_house_value'])

df_train_full, df_test = train_test_split(data, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.median_house_value.values
y_val = df_val.median_house_value.values
y_test = df_test.median_house_value.values

del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']

train_dict = df_train[cat + num].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
dv.fit(train_dict)

X_train = dv.transform(train_dict)

val_dict = df_val[cat + num].to_dict(orient='records')
X_val = dv.transform(val_dict)

for a in [0, 0.01, 0.1, 1, 10]:
    model = Ridge(alpha=a, solver="sag", random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    score = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(a, round(score, 3))

# 0 0.525
# 0.01 0.525
# 0.1 0.525
# 1 0.525
# 10 0.525

# 0






