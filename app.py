import numpy as np
from flask import request, jsonify, render_template, Flask
import pickle

# Init app + load model in read mode


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Route node where html should go 
@app.route('/')
def home():
    return render_template('index.html')

# Models use this route to predict
@app.route('/predict', methods=['POST'])
def predict():
    """ Gathers user inputs and puts inputs into array for model to predict i.e. Renders results on HTML GUI  """
    int_features = [int(x) for x in request.form.values()] # 3 inputs bc 3 features
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary is predicted to be ${}'.format(output))

# Static route that is passed hardcoded JSON to predict (data is gathered via request.py file)
@app.route('/predict_api', methods=['POST'])
def predict_app():
    """ For direct API calls through request """
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values))])

    output = prediction[0]
    return jsonify(output)

# Always one main fcn in flask
if __name__ == "__main__":
    app.run(debug=True)