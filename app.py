from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re

# Load the trained SVM model
with open('models/svm_spam_classifier.pkl', 'rb') as model_file:
    svm_model = pickle.load(model_file)

# Load the most common words
with open('models/most_common_words.pkl', 'rb') as words_file:
    most_common_words = pickle.load(words_file)

# Load the scaler used during training
with open('models/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize Flask application
app = Flask(__name__)

def preprocess_text(text):
    """Preprocess the email text by removing special characters and tokenizing."""
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()  # Remove non-alphabetic characters and lowercase
    words = text.split()  # Tokenize the text into words
    return words

def generate_word_count_vector(email_words, common_words):
    """Generate a word count vector for the email."""
    word_count_vector = [0] * len(common_words)
    for word in email_words:
        if word in common_words:
            index = common_words.index(word)
            word_count_vector[index] += 1
    return word_count_vector

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_email():
    email_text = request.form['email']  # Get email text from the form
    email_words = preprocess_text(email_text)  # Preprocess the text
    email_vector = generate_word_count_vector(email_words, most_common_words)  # Convert to vector
    email_vector = np.array(email_vector).reshape(1, -1)

    # Standardize the feature vector using the loaded scaler
    email_vector_scaled = scaler.transform(email_vector)

    # Predict with the model
    prediction = svm_model.predict(email_vector_scaled)

    if prediction[0] == 0:
        result = 'This is not a spam mail'
    else:
        result = 'This is a spam mail'

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(port=8000, debug=True)
