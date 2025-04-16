import pandas as pd
from flask import Flask, request, render_template
import pickle

# Load the Trained Model
with open('pipelined_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

# Index Route: Displays the form
@app.route('/')
def home():
    return render_template('index.html')  # This should be your form page

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract values from the form
        gender = request.form['gender']
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        smoking_history = request.form['smoking_history']
        bmi = float(request.form['bmi'])
        hba1c_level = float(request.form['HbA1c_level'])
        blood_glucose_level = int(request.form['blood_glucose_level'])

        # Create a DataFrame to feed to the model
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'smoking_history': [smoking_history],
            'bmi': [bmi],
            'HbA1c_level': [hba1c_level],
            'blood_glucose_level': [blood_glucose_level]
        })

        # Make prediction
        prediction = model.predict(input_data)
        result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

        return render_template('prediction.html', prediction_output=f'Prediction: {result}')

    except Exception as e:
        return render_template('prediction.html', prediction_output=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
