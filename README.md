# Sentiment Analysis with LSTM

This repository contains a sentiment analysis model built using TensorFlow and Keras. The model processes text reviews and predicts their sentiment (positive, negative, or neutral) using Long Short-Term Memory (LSTM) layers. The project includes the necessary preprocessing steps, model definition, training, evaluation, and prediction for new reviews.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Predictions](#predictions)
- [License](#license)

## Overview
The goal of this project is to classify text reviews into three categories:
- **Positive**
- **Negative**
- **Neutral**

We use an LSTM-based model, a type of Recurrent Neural Network (RNN), to analyze the text and predict the sentiment. The code also includes steps for preprocessing the text data, encoding labels, training the model, evaluating it on test data, and making predictions on new reviews.

## Requirements
Before running the code, make sure you have the following dependencies installed:
- TensorFlow (>= 2.0)
- Scikit-learn
- NumPy
- Keras
- Matplotlib (optional, for visualization)

You can install the necessary dependencies by running:
```
pip install tensorflow scikit-learn numpy matplotlib
```

## Setup Instructions
1. Clone this repository to your local machine:
   ```
   git clone https://github.com/your-username/sentiment-analysis-lstm.git
   ```

2. Navigate to the project directory:
   ```
   cd sentiment-analysis-lstm
   ```

3. Install the required dependencies (see above).

4. Modify the `reviews` and `labels` arrays in the code to include your dataset of reviews and sentiments.

5. Run the Python script to train the model:
   ```
   python sentiment_analysis.py
   ```

## Model Architecture
The model is built using the Keras Sequential API with the following architecture:
- **Embedding Layer**: Converts integer sequences into dense vectors of fixed size.
- **Bidirectional LSTM Layers**: Bidirectional LSTMs allow the model to capture information from both directions (past and future).
- **Dropout Layers**: Used to prevent overfitting by randomly dropping units during training.
- **Dense Layer**: A fully connected layer that outputs the final prediction with a softmax activation function.

## Training
The model is trained on the processed text data with the following hyperparameters:
- **Epochs**: 5
- **Batch Size**: 64
- **Validation Split**: 0.2 (20% of the training data is used for validation)

## Evaluation
After training, the model is evaluated on a separate test set, and the test loss and accuracy are printed.

## Predictions
Once the model is trained, it can predict the sentiment of new reviews. For example:
```python
new_review = ["This is the best product ever!"]
```
The prediction class will be printed as either "positive", "negative", or "neutral".
