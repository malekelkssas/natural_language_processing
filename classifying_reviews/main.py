import keras.preprocessing.text
from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
import os
import numpy as np


# 1-organizing data
# ____________________

vocab_size = 88584
Max_len = 250
batch_size = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)
# print(train_data[0])    #have a look at one review

# the len of the data review is not unique,so we must make all the data have the same length
# that for we will add some kind of padding like for example blank word

train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, Max_len)
test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data, Max_len)


# 2-creating the model
# _____________________
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size,32),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ]
)

model.summary()

# 3-training and testing
# _____________________
# after training the model once ,you can use (new_model) variable and comment the training part
# # ________________
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])
history = model.fit(train_data,train_labels,epochs=10,validation_split=0.2) #validation_split ->it means that I am going to use 20% of the data to evaluate


model.save("Sentiment Analysis.h5")
new_model = tf.keras.models.load_model("Sentiment Analysis.h5")
model = new_model
result = model.evaluate(test_data, test_labels)
print(result)

# 4-prediction
# __________
word_index = imdb.get_word_index()


def encode_text(text):  # preprocessing line texts
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]  # else 0 ->means we don't know what is this character or word
    return tf.keras.preprocessing.sequence.pad_sequences([tokens], Max_len)[0]  # it returns list of lists


text_test = "that movie was just amazing, so amazing"
encoded = encode_text(text_test)
# print(encoded)

# reversing the prev process

reverse_word_index = {value: key for (key, value) in word_index.items()}


def decode_integers(integres):
    pad = 0
    text = ""
    for num in integres:
        if num != pad:
            text += reverse_word_index[num] + " "

    return text[:-1]  # return everything except the last space


print(decode_integers(encoded))


def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))  # (1, Max_len)
    pred[0] = encoded_text
    pred_result = model.predict(pred)
    print(pred_result[0][0]*100,"% positive review")


positive = "that movie was so awesome! I really loved it and would watch it again because it was amazingly great"
predict(positive)

negative = "that movie sucked. I hated it and wouldn't watch it again. Was one of the worst things i have ever watched"
predict(negative)
