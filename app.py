
from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

data = pd.read_csv("heart.csv")
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

X = data[features]
y = data['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

@app.route('/')
def home():
    return render_template('index.html',accuracy=round(accuracy, 2))


@app.route('/predict', methods=['POST'])
def predict():

    try:
        values = []

        for feature in features:
            values.append(float(request.form[feature]))

        input_data = pd.DataFrame([values], columns=features)

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "Possible Heart Disease Detected"
        else:
            result = "No Heart Disease Detected"

        display_inputs = {
            "Age": request.form['age'] + " years",
            "Gender": "Male" if request.form['sex'] == "1" else "Female",

            "Chest Pain": {
                "0": "No chest pain",
                "1": "Typical chest pain",
                "2": "Atypical chest pain",
                "3": "Other non-anginal chest pain"
            }[request.form['cp']],

            "Resting Blood Pressure": request.form['trestbps'] + " mm Hg",
            "Cholesterol": request.form['chol'] + " mg/dL",

            "Fasting Blood Sugar Above 120": 
                "Yes" if request.form['fbs'] == "1" else "No",

            "Resting ECG": {
                "0": "Normal",
                "1": "ST-T wave abnormality",
                "2": "Left ventricular hypertrophy"
            }[request.form['restecg']],

            "Maximum Heart Rate": request.form['thalach'],

            "Chest Pain During Exercise":
                "Yes" if request.form['exang'] == "1" else "No",

            "ST Depression": request.form['oldpeak'],

            "Exercise ECG Slope": {
                "0": "Upsloping",
                "1": "Flat",
                "2": "Downsloping"
            }[request.form['slope']],

            "Major Blood Vessels": request.form['ca'],

            "Thalassemia": {
                "1": "Normal",
                "2": "Fixed defect",
                "3": "Reversible defect"
            }[request.form['thal']]
        }

        return render_template('result.html',result=result,inputs=display_inputs,accuracy=round(accuracy, 2))

    except (ValueError, KeyError):
        return "Invalid input. Please go back and enter valid values.", 400


if __name__ == '__main__':
    app.run(host='0.0.0.0',
            port=int(os.environ.get('PORT', 10000)))