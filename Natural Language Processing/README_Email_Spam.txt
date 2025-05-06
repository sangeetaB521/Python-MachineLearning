
# Email Spam Detection Using Naive Bayes Classifier

This project implements an email spam detection system using a Naive Bayes Classifier. It leverages text preprocessing and vectorization techniques to classify email messages as spam or not spam.

## Overview

Spam emails are a persistent problem, and building an effective spam detection system is crucial for reducing unnecessary clutter in email inboxes. This project focuses on developing a machine learning model using Python to detect spam emails based on their content.

## Dataset

The dataset used in this project contains email texts labeled as spam (1) or not spam (0). It is assumed to be pre-processed and loaded into a Pandas DataFrame.

## Features

- Text vectorization using CountVectorizer
- Implementation of Multinomial Naive Bayes for classification
- Evaluation of the model using accuracy metrics
- Testing the model with custom email samples

## Requirements

The following Python libraries are required to run the project:
- pandas
- sklearn

Install the required packages using pip:

```bash
pip install pandas scikit-learn
```

## Code Details

### Text Vectorization

The email text is converted into a matrix of token counts using the `CountVectorizer` from `sklearn.feature_extraction.text`.

### Model Training

The `MultinomialNB` classifier from `sklearn.naive_bayes` is used to train the model on the vectorized text data.

### Testing

Custom email samples are provided to the trained model to predict whether they are spam or not.

## Sample Code

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])

# Training
model = MultinomialNB()
model.fit(X, df['spam'])

# Prediction
samples = ["Free money!!!", "Please reach out when free"]
sample_vectors = vectorizer.transform(samples)
predictions = model.predict(sample_vectors)
print(predictions)
```

## Results

The Naive Bayes Classifier provides a simple yet effective approach to spam detection. The accuracy and performance depend on the quality and size of the dataset used.

## License

This project is released under the MIT License.

