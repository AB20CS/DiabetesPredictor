from lzma import PRESET_DEFAULT
from flask import Flask, request, render_template
import pandas as pd
import joblib


# Declare a Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = joblib.load("clf.pkl")
        
        # Get values through input bars
        Glucose = request.form.get("Glucose")
        BloodPressure = request.form.get("BloodPressure")
        SkinThickness = request.form.get("SkinThickness")
        Insulin = request.form.get("Insulin")
        BMI = request.form.get("BMI")
        DiabetesPedigreeFunction = request.form.get("DiabetesPedigreeFunction")
        Age = request.form.get("Age")
        
        # Put inputs to dataframe
        X = pd.DataFrame([[Glucose, BloodPressure, SkinThickness, Insulin, BMI,
                           DiabetesPedigreeFunction, Age]], 
                           columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                                      'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Get prediction
        prediction = clf.predict(X)[0]

        if prediction == 0:
            prediction = 'Not Diabetic'
        else:
            prediction = 'Diabetic'

        
    else:
        prediction = ""
        
    return render_template("website.html", output = prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug = True)