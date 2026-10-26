# IMDB sentiment analysis using Embedding + LSTM (same style as my Food11 CNN project)
import numpy as np
import matplotlib.pyplot as plt

## Load IMDB Dataset (reviews are already converted to word IDs)
from tensorflow.keras.datasets import imdb

# Keep only the 10000 most used words (vocabulary)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

## Check Data
print("Train reviews :", len(X_train))
print("Test reviews :", len(X_test))
print("First review IDs :", X_train[0][:10])
print("First review label :", y_train[0])
print("First review length :", len(X_train[0]))

## Padding (make every review exactly 100 word IDs)
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train = pad_sequences(X_train, maxlen=100)
X_test = pad_sequences(X_test, maxlen=100)

print("X_train shape :", X_train.shape)
print("X_test shape :", X_test.shape)

## Create Model
from tensorflow.keras.models import Sequential

model = Sequential()

## Import Layers
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

# Input Layer (each review = 100 word IDs after padding)
model.add(Input(shape=(100,)))

# Embedding Layer (10000 words in vocabulary, each word -> 32 numbers)
model.add(Embedding(10000, 32))

# LSTM Layer
model.add(LSTM(32))

# Output Layer (positive / negative)
model.add(Dense(1, activation='sigmoid'))

## Compile Model
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

## Model Summary
model.summary()

## Model Train
history = model.fit(
    X_train, y_train,
    epochs = 5,
    batch_size = 64,
    validation_split = 0.2
)

## Accuracy Curve (check overfitting)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('imdb_accuracy_curve.png')
plt.show()

## Loss Curve (check overfitting)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('imdb_loss_curve.png')
plt.show()

## Evaluate Model On Test Data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy :", test_accuracy)

## Prediction (probability between 0 and 1)
y_prob = model.predict(X_test)

## Probability -> Class (0.5 threshold)
y_pred = (y_prob > 0.5).astype(int)

## Compare Actual VS Predicted
print("Actual Classes :", y_test[:10])
print("Predicted Classes :", y_pred[:10].flatten())

## Classification Report
from sklearn.metrics import classification_report

cr = classification_report(y_test, y_pred, target_names=['negative', 'positive'])
print("Classification Report :\n", cr)

## Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :\n", cm)

## Model Save
model.save('imdb_lstm_model.keras')
