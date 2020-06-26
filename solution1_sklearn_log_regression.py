import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

#Load training dataset
df_train = pd.read_csv('train.csv', delimiter = ',')
x_data = df_train["text"].values
Y = df_train["target"].values

#Split into training and validation
x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x_data, Y, test_size=0.2, random_state=7)

#Create, train and validate model
model = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression())])
model.fit(x_train, y_train)
pred_train = model.predict(x_train)
print('training: %.2f' % np.mean(pred_train == y_train))
pred_val = model.predict(x_validation)
print('validation: %.2f' % np.mean(pred_val == y_validation))

#Apply model to test dataset
df_test = pd.read_csv('test.csv', delimiter = ',')
pred_test = model.predict(df_test['text'].values)
print('test: %.2f' % np.mean(pred_test == df_test['target'].values))

