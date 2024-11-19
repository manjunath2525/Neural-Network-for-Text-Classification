import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    GlobalAveragePooling1D,
    BatchNormalization
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Sample dataset (Add more reviews for better accuracy)
reviews = [
    "I love this product!",
    "This is a bad product.",
    "It is okay.",
    "Amazing quality and great service.",
    "Worst purchase I have ever made.",
    "Highly recommend this!",
    "Not worth the price.",
    "Very satisfied with my purchase.",
    "It didn't work as expected.",
    "Great product for the price.",
]

# Sentiments (replace with actual sentiment labels from your dataset)
labels = ["positive", "negative", "neutral", "positive", "negative", "positive", "negative", "positive", "negative", "positive"]

# Data preprocessing
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(reviews)

X = tokenizer.texts_to_sequences(reviews)
X = pad_sequences(X, maxlen=200)

le = LabelEncoder()
y = le.fit_transform(labels)

# Split the data into training and testing sets (more balanced split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(GlobalAveragePooling1D())
model.add(BatchNormalization())  # Batch normalization to stabilize learning
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(3, activation="softmax"))

# Compile the model with a lower learning rate and use class weights
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

# Train the model with early stopping and class weights
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# Class weights for handling class imbalance (if any)
class_weights = {0: 1.0, 1: 2.0, 2: 1.0}  # Adjust weights depending on class distribution

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, 
          callbacks=[early_stopping], class_weight=class_weights)

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

# Evaluate model with classification report and confusion matrix
y_pred = model.predict(X_test)
y_pred_class = y_pred.argmax(axis=1)

print("Classification Report:\n", classification_report(y_test, y_pred_class))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_class))
