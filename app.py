from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Define columns to encode and scale
categorical_columns = [
    "Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"
]
numerical_columns = ["Compartments", "Weight Capacity (kg)"]

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to preprocess input dynamically
def preprocess_input(data):
    processed_data = []
    encoders = {}  # Create fresh encoders for this input batch

    # Fit and transform categorical columns
    for col in categorical_columns:
        encoder = LabelEncoder()
        encoder.fit([data[col]])  # Fit only on the provided input
        processed_data.append(encoder.transform([data[col]])[0])  # Transform single value
        encoders[col] = encoder

    # Add numerical columns
    numerical_data = [data[col] for col in numerical_columns]
    processed_data.extend(numerical_data)

    # Scale numerical data
    scaler = StandardScaler()
    numerical_data = np.array(numerical_data).reshape(1, -1)
    scaler.fit(numerical_data)  # Fit scaler dynamically
    scaled_numerical_data = scaler.transform(numerical_data)[0]

    # Combine categorical and scaled numerical data
    processed_data[len(categorical_columns):] = scaled_numerical_data

    return np.array(processed_data).reshape(1, -1)

# Routes
@app.route('/')
def home():
    return render_template('home.html')  # Home page template

@app.route('/data_entry', methods=['GET', 'POST'])
def data_entry():
    if request.method == 'POST':
        # Retrieve form data
        try:
            data = {
                "Brand": request.form.get("Brand"),
                "Material": request.form.get("Material"),
                "Size": request.form.get("Size"),
                "Compartments": float(request.form.get("Compartments")),
                "Laptop Compartment": request.form.get("Laptop_Compartment"),
                "Waterproof": request.form.get("Waterproof"),
                "Style": request.form.get("Style"),
                "Color": request.form.get("Color"),
                "Weight Capacity (kg)": float(request.form.get("Weight_Capacity")),
            }
        except ValueError as e:
            return f"Invalid input data: {str(e)}", 400
        
        # Redirect to save_data with raw data
        return redirect(url_for('save_data', **data))
    return render_template('data.html')  # Form page template

@app.route('/save_data', methods=['GET', 'POST'])
def save_data():
    # Retrieve data from query parameters
    data = request.args.to_dict()

    # Convert numerical fields back to float
    #data["Compartments"] = data["Compartments"].astype("float")
    #data["Weight Capacity (kg)"] = float(data["Weight Capacity (kg)"])

    # Display saved data
    return render_template('save_data.html', data=data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve data from the form submission
        data = {
            "Brand": request.form.get("Brand"),
            "Material": request.form.get("Material"),
            "Size": request.form.get("Size"),
            "Compartments": (request.form.get("Compartments")),
            "Laptop Compartment": request.form.get("Laptop_Compartment"),
            "Waterproof": request.form.get("Waterproof"),
            "Style": request.form.get("Style"),
            "Color": request.form.get("Color"),
            "Weight Capacity (kg)": (request.form.get("Weight Capacity")),
        }

        # Preprocess input data
        processed_data = preprocess_input(data)

        # Predict using the pre-trained model
        prediction = model.predict(processed_data)[0]

        return render_template('predict.html', prediction=prediction)
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
