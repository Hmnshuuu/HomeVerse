from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
import sklearn

app = Flask(__name__)

with open('house_price_prediction_model.joblib', 'rb') as file:
    model = joblib.load(file)

@app.route('/')
def home():
    return render_template('index2.html')


@app.route('/predict', methods=['POST','GET'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    print(input_data)
    input_df = pd.DataFrame([input_data], columns=['Avg. Area Income', 'Avg. Area House Age', 
                                                   'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 
                                                   'Area Population'])
    print(input_df)
    # check scikit-learn version
    if sklearn.__version__ != '1.0.2':
        print(f"Warning: Model was trained with scikit-learn version 1.0.2 but the current version is {sklearn.__version__}. Results may not match the expected outputs.")
    
    predicted_price = model.predict(input_df)[0]
    predicted_price = round(predicted_price, 2)
    print(predicted_price)
    return render_template('index2.html', prediction_text='Predicted Price: $' + str(predicted_price))

if __name__ == '__main__':
    app.run(debug=True)
