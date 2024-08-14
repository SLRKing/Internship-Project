from flask import Flask, request, render_template
import pandas as pd
import pickle

# Load the trained model, scaler, and label encoders
model = pickle.load(open('best_credit_risk_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))

app = Flask(__name__)

# Route for handling the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling form submission and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetching input data from form submission
        data = []
        for feature in ['status', 'duration', 'credit_history', 'purpose', 'amount', 'savings',
                        'employment_duration', 'installment_rate', 'personal_status_sex',
                        'other_debtors', 'present_residence', 'property', 'age',
                        'other_installment_plans', 'housing', 'number_credits', 'job',
                        'people_liable', 'telephone', 'foreign_worker']:
            value = request.form[feature]
            if feature in label_encoders:  # Convert categorical features using label encoders
                value = label_encoders[feature].transform([value])[0]
            data.append(float(value))

        # Convert the list to a DataFrame
        features = ['status', 'duration', 'credit_history', 'purpose', 'amount', 'savings',
                    'employment_duration', 'installment_rate', 'personal_status_sex',
                    'other_debtors', 'present_residence', 'property', 'age',
                    'other_installment_plans', 'housing', 'number_credits', 'job',
                    'people_liable', 'telephone', 'foreign_worker']
        
        input_data = pd.DataFrame([data], columns=features)

        # Standardize the input data
        input_data = scaler.transform(input_data)

        # Making prediction using the loaded model
        prediction = model.predict(input_data)[0]

        # Mapping the prediction result
        if prediction == 1:
            result = 'Good Credit Risk'
        else:
            result = 'Bad Credit Risk'

        # Displaying the result in output.html
        return render_template('output.html', prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
