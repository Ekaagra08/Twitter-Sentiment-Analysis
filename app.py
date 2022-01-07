# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the classification model and tf-idf Vectorizer object from disk
classifier = pickle.load(open('ts_model.pkl', 'rb'))
vectorizer  = pickle.load(open('vectorize-transform.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = vectorizer .transform(data).toarray()
    	my_prediction = classifier.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)