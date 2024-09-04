from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('final_xgboost_model.pkl')

# Define the list of required features
required_features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',
       'GrLivArea', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
       'MSZoning', 'Neighborhood', 'BsmtQual', 'BsmtFinType1', 'CentralAir',
       'GarageType']



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.form.to_dict()
    
    # Ensure all required features are in the data
    for feature in required_features:
        if feature not in data:
            data[feature] = 0  # or any other default value appropriate for your context
    
    # Create DataFrame with the correct order
    df = pd.DataFrame([data], columns=required_features)
    
    # Convert categorical columns to the same format as used during training
    
    
    # Predict using the loaded model
    prediction = model.predict(df)
    
    # Return prediction as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
