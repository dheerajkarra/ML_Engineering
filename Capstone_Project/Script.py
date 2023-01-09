# Imported Libraries
# from sklearn.ensemble import RandomForestClassifier

import numpy as np
from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# Change directory to working directory
import os
os.chdir('F:\\Projects\\ML_Engineering\\Capstone_Project\\')

# Load Data
df = pd.read_csv('winemag-data_first150k.csv')

counter = Counter(df['variety'].tolist())
top_10_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}
df = df[df['variety'].map(lambda x: x in top_10_varieties)]

print(df.head())

####################################################################
# EDA
####################################################################

# df.describe()

df = df.drop(['Unnamed: 0'],axis = 1)
# shape of the dataset
print(df.shape) # (85520, 10)

# total number of unique categories
print("Unique categories:",df['variety'].nunique()) # 10
print("-------------------------------------------------")
# information about metadata
df.info()


# produce pandas profiling report - Automated EDA
# import pandas_profiling as pp
# df.profile_report()

###################################################################
# Visualizations of EDA
###################################################################

# # Top categories by and number of articles per categories
# cat_df = pd.DataFrame(df['variety'].value_counts()).reset_index()
# cat_df.rename(columns={'index':'variety','variety':'numcat'}, inplace=True)

# # Visualize top 10 categories and proportion of each categories in dataset
# plt.figure(figsize=(10,6))
# ax = sns.barplot(data = cat_df, x = np.array(cat_df.variety)[:10], y = np.array(cat_df.numcat)[:10])
# # ax = sns.barplot( np.array(cat_df.numcat)[:10])
# for p in ax.patches:
#     ax.annotate(p.get_height(), (p.get_x()+0.01, p.get_height() + 50))
# plt.title("TOP 10 Categories", size=15)
# plt.xlabel("Categories", size=14)
# plt.xticks(rotation=45)
# plt.ylabel("Number of rows", size=14)
# plt.show()


# # plot the pie chart of top 10 provinces
# cat_df = pd.DataFrame(df['province'].value_counts()).reset_index()
# cat_df.rename(columns={'index':'province','province':'numcat'}, inplace=True)

# fig = plt.figure(figsize=(12,12))
# A = plt.pie(cat_df['numcat'][:10],
#             labels=cat_df['province'][:10],
#             autopct='%1.1f%%',
#             startangle=90,
#             labeldistance=1.08,
#             pctdistance=1.03,
#             rotatelabels=45
#             )

# plt.title("Pie Chart of TOP 10 provinces", size=20, weight='bold')
# plt.show()


# # wordcloud of varieties in our dataset

# plt.figure(figsize=(12,12))
# wc = WordCloud(max_words=1000, 
#                 min_font_size=10,
#                 height=600,
#                 width=1600,
#                 background_color='black',
#                 contour_color='black',
#                 colormap='plasma',
#                 repeat=False,
#                 stopwords=STOPWORDS).generate(' '.join(df.description))

# plt.title("All description Wordcloud", size=15, weight='bold')
# plt.imshow(wc, interpolation= "bilinear")
# plt.axis('off')


# # create new dataframe 
# ndf = df.copy()

# # list of top 10 categories in out dataset
# categories = df['variety'].unique()

# # list of top 10 categories list
# cat_list = []

# for i in categories:
#     cat_ndf = ndf[ndf['variety'] == i]
#     cat_array = cat_ndf['description'].values  # array of news articles text in each category
#     cat_list.append(cat_array)
    
# # create a wordcloud instance
# wc1 = WordCloud(max_words=1000, 
#                 min_font_size=10,
#                 height=600,
#                 width=1600,
#                 background_color='black',
#                 contour_color='black',
#                 colormap='plasma',
#                 repeat=True,
#                 stopwords=STOPWORDS)

# # plot the figure of 10 wordcloud from out dataset
# plt.figure(figsize=(15,15))

# for idx, j in enumerate(categories):
#     plt.subplot(5,2,idx+1)
#     cloud = wc1.generate(' '.join(cat_list[idx]))
#     plt.imshow(cloud, interpolation= "bilinear")
#     plt.title(f"Wordcloud for category {j}")
#     plt.axis('off')

###################################################################
# Text pre-processing
###################################################################

# Function for cleaning the text
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# clean the text data using regex and data cleaning function
def datacleaning(text):
    whitespace = re.compile(r"\s+")
    user = re.compile(r"(?i)@[a-z0-9_]+")
    text = whitespace.sub(' ', text)
    text = user.sub('', text)
    text = re.sub(r"\[[^()]*\]","", text)
    text = re.sub("\d+", "", text)
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r"(?:@\S*|#\S*|http(?=.*://)\S*)", "", text)
    text = text.lower()
    
    # removing stop-words
    text = [word for word in text.split() if word not in list(STOPWORDS)]
    
    # word lemmatization
    sentence = []
    for word in text:
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word,'v'))
        
    return ' '.join(sentence)


# Example of pre-processing using above function
print("Text sentence before pre-processing:\n",df['description'][0])
print("---"*35)
print("Text sentence after pre-processing:\n",datacleaning(df['description'][0]))


# Text sentence before pre-processing:
#  This tremendous 100% varietal wine hails from Oakville and was aged over three years in oak. Juicy red-cherry fruit and a compelling hint of caramel greet the palate, framed by elegant, fine tannins and a subtle minty tone in the background. Balanced and rewarding from start to finish, it has years ahead of it to develop further nuance. Enjoy 2022–2030.
# ---------------------------------------------------------------------------------------------------------
# Text sentence after pre-processing:
#  tremendous varietal wine hail oakville age three years oak juicy redcherry fruit compel hint caramel greet palate frame elegant fine tannins subtle minty tone background balance reward start finish years ahead develop nuance enjoy


###################################################################
# Encoding to numerical form
###################################################################

# Cleaning and modelling
df['description_clean'] = df['description'].map(lambda x: datacleaning(x))
description_list = df['description_clean'].tolist()
varietal_list = [top_10_varieties[i] for i in df['variety'].tolist()]
varietal_list = np.array(varietal_list)

count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(description_list)

tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

train_x, test_x, train_y, test_y = train_test_split(x_train_tfidf, varietal_list, test_size=0.3)

###################################################################
# Modelling
###################################################################

# Naive Bayes
clf = MultinomialNB().fit(train_x, train_y)
#clf = SVC(kernel='linear').fit(train_x, train_y)
y_score = clf.predict(test_x)

n_right = 0
for i in range(len(y_score)):
    if y_score[i] == test_y[i]:
        n_right += 1

print("Accuracy: %.2f%%" % ((n_right/float(len(test_y)) * 100)))

# Accuracy: 64.99%

###################################################################
# Features
###################################################################

# Functions to get the top words
from nltk import word_tokenize
from collections import defaultdict

def count_top_x_words(corpus, top_x, skip_top_n):
    count = defaultdict(lambda: 0)
    for c in corpus:
        for w in word_tokenize(c):
            count[w] += 1
    count_tuples = sorted([(w, c) for w, c in count.items()], key=lambda x: x[1], reverse=True)
    return [i[0] for i in count_tuples[skip_top_n: skip_top_n + top_x]]


def replace_top_x_words_with_vectors(corpus, top_x):
    topx_dict = {top_x[i]: i for i in range(len(top_x))}

    return [
        [topx_dict[w] for w in word_tokenize(s) if w in topx_dict]
        for s in corpus
    ], topx_dict


def filter_to_top_x(corpus, n_top, skip_n_top=0):
    top_x = count_top_x_words(corpus, n_top, skip_n_top)
    return replace_top_x_words_with_vectors(corpus, top_x)

###################################################################
# Neural network DL approach using Keras
###################################################################

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
# from keras.layers.embeddings import Embedding
from keras.layers import Embedding
# from keras.preprocessing import sequence
from keras.utils import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
#from lib.get_top_xwords import filter_to_top_x

# df = pd.read_csv('data/wine_data.csv')
# df = pd.read_csv('/kaggle/input/wine-reviews/winemag-data_first150k.csv')

# counter = Counter(df['variety'].tolist())
# top_10_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(10))}
# df = df[df['variety'].map(lambda x: x in top_10_varieties)]

# df['description_clean'] = df['description'].map(lambda x: datacleaning(x))
description_list = df['description_clean'].tolist()

# description_list = df['description'].tolist()
mapped_list, word_list = filter_to_top_x(description_list, 2500, 10)
varietal_list_o = [top_10_varieties[i] for i in df['variety'].tolist()]
varietal_list = to_categorical(varietal_list_o)

max_review_length = 150

mapped_list = pad_sequences(mapped_list, maxlen=max_review_length)
train_x, test_x, train_y, test_y = train_test_split(mapped_list, varietal_list, test_size=0.3)

max_review_length = 150

embedding_vector_length = 64
model = Sequential()

model.add(Embedding(2500, embedding_vector_length, input_length=max_review_length))
model.add(Conv1D(50, 5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(max(varietal_list_o) + 1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=5, batch_size=64)

y_score = model.predict(test_x)
y_score = [[1 if i == max(sc) else 0 for i in sc] for sc in y_score]
n_right = 0
for i in range(len(y_score)):
    if all(y_score[i][j] == test_y[i][j] for j in range(len(y_score[i]))):
        n_right += 1

print("Accuracy: %.2f%%" % ((n_right/float(len(test_y)) * 100)))

# Epoch 1/5
# 936/936 [==============================] - 23s 23ms/step - loss: 0.9859 - accuracy: 0.6609
# Epoch 2/5
# 936/936 [==============================] - 21s 23ms/step - loss: 0.5866 - accuracy: 0.8042
# Epoch 3/5
# 936/936 [==============================] - 20s 22ms/step - loss: 0.4720 - accuracy: 0.8411
# Epoch 4/5
# 936/936 [==============================] - 22s 23ms/step - loss: 0.3599 - accuracy: 0.8816
# Epoch 5/5
# 936/936 [==============================] - 22s 23ms/step - loss: 0.2502 - accuracy: 0.9198
# Accuracy: 82.99%


model.save('text_clf.h5')


