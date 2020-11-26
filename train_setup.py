import jsonlines
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# https://towardsdatascience.com/tensorflow-sarcasm-detection-in-20-mins-b549311b9e91

def read_train_file():
    labels = []
    responses = []
    with jsonlines.open('data/train.jsonl') as f:
        for line in f.iter():
            labels.append(line['label'])
            responses.append(line['response'].replace('@USER', '').strip())
    return labels, responses

def read_test_file():
    responses = []
    with jsonlines.open('data/test.jsonl') as f:
        for line in f.iter():
            responses.append(line['response'].replace('@USER', '').strip())
    return responses

def splitter(labels, responses):
    train_labels = []
    test_labels = []
    train_responses = []
    test_responses = []
    for i in range(0, len(responses) / 2):
        if (i % 2 == 0):
            train_labels.append(labels[i])
            train_responses.append(responses[i])
            test_labels.append(labels[i+ len(responses) / 2])
            test_responses.append(responses[i+ len(responses) / 2] )
        else:
            train_labels.append(labels[i+ len(responses) / 2])
            train_responses.append(responses[i+ len(responses) / 2])
            test_labels.append(labels[i])
            test_responses.append(responses[i])
    return train_labels, test_labels, train_responses, test_responses

labels, responses = read_train_file()
for i in range(0, len(labels)):
    if (labels[i] == "SARCASM"):
        labels[i] = 1
    else :
        labels[i] = 0
train_labels, test_labels, train_responses, test_responses = splitter(labels, responses)

vocab_size = 10000
oov_tok = "<oov>"

# Fit the tokenizer on Training data
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_responses)

word_index = tokenizer.word_index
# Setting the padding properties
max_length = 100
trunc_type='post'
padding_type='post'

# Creating padded sequences from train and test data
training_sequences = tokenizer.texts_to_sequences(train_responses)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(test_responses)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


# Setting the model parameters
embedding_dim = 16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# model.summary()

# Converting the lists to numpy arrays for Tensorflow 2.x
training_padded = np.array(training_padded)
training_labels = np.array(train_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(test_labels)


# Training the model
num_epochs = 30
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)
# result = []
outlier = 0
test_new_responses = read_test_file()
for i in range(0, len(test_new_responses)):
    sequences = tokenizer.texts_to_sequences(test_new_responses[i])
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    result = model.predict(padded)
    if (result[0] == result[1]):
        outlier += 1
    # result.append(model.predict(padded))

print(outlier)