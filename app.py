from flask import Flask, render_template, request
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
model = pickle.load(open('Spam SMS Detection.pickle', 'rb'))
vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))  # Load the fitted vectorizer

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        transformed_data = vectorizer.transform(data)  # Use the fitted vectorizer
        prediction = model.predict(transformed_data)[0]
        output = 'Spam' if prediction == 1 else 'Not Spam'
        return render_template('index.html', prediction_text=f'This message is: {output}')

if __name__ == '__main__':
    app.run(debug=True)
