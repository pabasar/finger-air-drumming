# Import necessary libraries
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# Function to load data from a file
def load_data(file_path):
    # Load and return data from pickle file
    data_dict = pickle.load(open(file_path, 'rb'))
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])
    return data, labels

# Function to split data into training and testing sets
def split_data(data, labels, test_size=0.2):
    # Split data and return
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, shuffle=True, stratify=labels)
    return x_train, x_test, y_train, y_test

# Function to train a Gradient Boosting Classifier
def train_model(x_train, y_train):
    # Initialize model, fit on training data, and return
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    return model

# Function to evaluate model performance
def evaluate_model(model, x_test, y_test):
    # Predict on testing data and calculate accuracy
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_predict)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    # Print classification report
    print("Classification Report:\n", classification_report(y_test, y_predict))

    # Return accuracy score
    return score

# Function to save model to a file
def save_model(model, file_path):
    # Save model to pickle file
    with open(file_path, 'wb') as f:
        pickle.dump({'model': model}, f)

# Load data
data, labels = load_data('./data.pickle')

# Split data
x_train, x_test, y_train, y_test = split_data(data, labels)

# Train model
model = train_model(x_train, y_train)

# Evaluate model
score = evaluate_model(model, x_test, y_test)

# Print accuracy
print('Classification accuracy:', score * 100)

# Save model
save_model(model, 'model.p')
