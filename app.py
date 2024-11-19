import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# Load the dataset
reviews = ["I love this product!", "This is a bad product.", "It is okay."]

# Define sentiments (replace with actual sentiments from your dataset)
labels = ["positive", "negative", "neutral"]

# Data preprocessing
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(reviews)

X = tokenizer.texts_to_sequences(reviews)
X = pad_sequences(X, maxlen=200)

le = LabelEncoder()
y = le.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model architecture
model = Sequential()
model.add(
    Embedding(input_dim=5000, output_dim=128, input_length=200)
)  # Adjust input_dim as needed
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(
    Bidirectional(LSTM(64))
)  # Adjust the number of units based on performance
model.add(Dropout(0.5))
model.add(Dense(3, activation="softmax"))

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Predict on new reviews
new_review = ["This is the best product ever!"]
new_sequence = tokenizer.texts_to_sequences(new_review)
new_padded = pad_sequences(new_sequence, maxlen=200)
pred = model.predict(new_padded)

# Get the predicted class
pred_class = le.inverse_transform([pred.argmax()])
print(f"Predicted class: {pred_class[0]}")