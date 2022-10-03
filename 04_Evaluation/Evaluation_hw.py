import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('F:\\Projects\\ML_Engineering\\04_Evaluation\\')

# Credit card data
df = pd.read_csv('credit_card_data.csv')

df['card_target'] = 1*(df['card']=='yes')
df[['card','card_target']].value_counts()


from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.card_target.values
y_val = df_val.card_target.values
y_test = df_test.card_target.values

del df_train['card_target']
del df_val['card_target']
del df_test['card_target']

###################################################################
# Question 1

num_cols = list(df.dtypes[df.dtypes!='object'].index)
num_cols.remove('card_target')

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

for c in num_cols:
    auc = roc_auc_score(y_train, df_train[c])
    if auc < 0.5:
        auc = roc_auc_score(y_train, -df_train[c])
    print('%9s, %.3f' % (c, auc))

#  reports, 0.717
#       age, 0.524
#    income, 0.591
#     share, 0.989
# expenditure, 0.991
# dependents, 0.533
#    months, 0.529
# majorcards, 0.534
#    active, 0.604


# share is the highest among the options

###################################################################
# Question 2

sel_cols = ["reports", "age", "income", "share", "expenditure", "dependents",
            "months", "majorcards", "active", "owner", "selfemp"]

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

train_dicts = df_train[sel_cols].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

val_dicts = df_val[sel_cols].to_dict(orient='records')
X_val = dv.transform(val_dicts)

y_pred = model.predict_proba(X_val)[:, 1]

round(roc_auc_score(y_val, y_pred),3) # 0.995

y_pred_bin = model.predict(X_val)
round(roc_auc_score(y_val, y_pred_bin),3) # 0.974

###################################################################
# Question 3

def confusion_matrix_dataframe(y_val, y_pred):
    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)
    
    return df_scores

df_scores = confusion_matrix_dataframe(y_val, y_pred)
df_scores[::10]

df_scores['p'] = df_scores.tp / (df_scores.tp + df_scores.fp)
df_scores['r'] = df_scores.tp / (df_scores.tp + df_scores.fn)
plt.plot(df_scores.threshold, df_scores.p, label='precision')
plt.plot(df_scores.threshold, df_scores.r, label='recall')

plt.vlines(0.24, 0, 1, color='grey', linestyle='--', alpha=0.5)

plt.legend()
plt.show()

# df_scores[df_scores['p']==df_scores['r']]
# Out[55]: 
#     threshold   tp  fp  fn  tn         p         r
# 24       0.24  206   5   5  48  0.976303  0.976303
# 25       0.25  206   5   5  48  0.976303  0.976303
# 26       0.26  206   5   5  48  0.976303  0.976303
# 27       0.27  206   5   5  48  0.976303  0.976303
# 28       0.28  206   5   5  48  0.976303  0.976303
# 29       0.29  206   5   5  48  0.976303  0.976303

# df_scores[df_scores['threshold']==0.3]
# Out[56]: 
#     threshold   tp  fp  fn  tn        p         r
# 30        0.3  205   5   6  48  0.97619  0.971564


# threshold = 0.3 for intersection

###################################################################
# Question 4

df_scores['f1'] = 2 * df_scores.p * df_scores.r / (df_scores.p + df_scores.r)
plt.figure(figsize=(10, 5))

plt.plot(df_scores.threshold, df_scores.f1)
plt.vlines(0.4, 0, 1, color='grey', linestyle='--', alpha=0.5)

plt.xticks(np.linspace(0, 1, 11))
plt.show()

# df_scores[df_scores['f1'] == df_scores['f1'].max()]
# Out[59]: 
#     threshold   tp  fp  fn  tn         p         r        f1
# 35       0.35  205   1   6  52  0.995146  0.971564  0.983213
# 36       0.36  205   1   6  52  0.995146  0.971564  0.983213
# 37       0.37  205   1   6  52  0.995146  0.971564  0.983213
# 38       0.38  205   1   6  52  0.995146  0.971564  0.983213
# 39       0.39  205   1   6  52  0.995146  0.971564  0.983213
# 40       0.40  205   1   6  52  0.995146  0.971564  0.983213
# 41       0.41  205   1   6  52  0.995146  0.971564  0.983213

# 0.4 threshold - f1 is maximum

###################################################################
# Question 5

from sklearn.model_selection import KFold

def train(df_train, y_train, C=1.0):
    dicts = df_train[sel_cols].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[sel_cols].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.card_target.values
    y_val = df_val.card_target.values

    dv, model = train(df_train, y_train, C=1.0)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('%.3f +- %.3f' % (np.mean(scores), np.std(scores))) # 0.996 +- 0.003
# sd = 0.003

###################################################################
# Question 6

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for C in [0.01, 0.1, 1, 10]:
    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.card_target.values
        y_val = df_val.card_target.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%4s, %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

# C=0.01, 0.992 +- 0.006
# C= 0.1, 0.995 +- 0.004
# C=   1, 0.996 +- 0.003
# C=  10, 0.996 +- 0.003

# C = 1 as it is smallest


