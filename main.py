import re
import json
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.src.utils.data_utils import pad_sequences
from keras.layers import Dense, Embedding, LSTM, Dropout, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords

# data preprocessing
data = pd.read_csv('news.csv')
print(data)
print(data.shape)
print(data.head(17))
print(data['label'].value_counts(normalize=True))
sns.countplot(x=data['label'])

data.dropna(axis=0) # Remove empty lines from csv file

# Load json file
with open('examples.json') as json_parent_variable:
  fake_news_dict = json.load(json_parent_variable)

# data preprocessing
def clean_data(string):
    result = re.sub('\n', '', string)                     # remove new lines
    result = re.sub('s+', '', string)                     # remove HTML tags
    result = re.sub(r'[^A-Za-z0-9 ]+', '', result)        # remove non-alphanumeric characters
    result = result.lower()
    return result

data = data.drop(columns=['title'], axis=1)
data['text'] = data['text'].apply(lambda cw: clean_data(str(cw)))
X = data['text']
print(data.head(7))

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
print(data.head(7))

# convert the labels to numeric form
class_label = data.label.factorize()
print(class_label)

# tokenize words (convert text into an array of vector embeddings)
tweet = data.text.values
t = Tokenizer(num_words=50000)    # initializing the tokenizer
t.fit_on_texts(tweet)             # fitting on the text data
vocab_size = len(t.word_index) + 1
encoded_mess = t.texts_to_sequences(tweet)   # creating the numerical sequence

# look into some text and corresponding numerical sequence
for i in range(5):
    print("Text               : ", X[i])
    print("Numerical Sequence : ", encoded_mess[i])

# use padding to pad the sentences to have equal length
max_len_seq = 100
padded_sequence = pad_sequences(encoded_mess, maxlen=max_len_seq)

# build the text classifier
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_len_seq))
model.add(SpatialDropout1D(0.5))  # use dropout mechanism
model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.25))
model.add(Dropout(0.4))
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# train the model with our data
history = model.fit(padded_sequence, class_label[0], validation_split=0.2, epochs=10, batch_size=128)

# do some predictions
def fake_news_detect(text):
    print("\n" + text)
    tw = t.texts_to_sequences([text])
    tw = pad_sequences(tw, maxlen=max_len_seq)
    padded_sequence = pad_sequences(tw, maxlen=max_len_seq)
    prediction = int(model.predict(padded_sequence).round().item())
    print("Predicted label: ", class_label[1][prediction], "\n")

for key in fake_news_dict:
    fake_news_detect(fake_news_dict[key]['value'])