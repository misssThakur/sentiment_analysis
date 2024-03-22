from flask import Flask, render_template, request
import joblib

# Initialize Flask app
app = Flask(__name__)

# Define routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        rf_model = joblib.load("model/naive_bayes.pkl")
        prediction = rf_model.predict([review])[0]
        if prediction == 1:
            sentiment = 'Positive'
        else:
            sentiment = 'Negative'
        return render_template('output.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")