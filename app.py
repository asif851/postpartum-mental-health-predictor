from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# Global variables
anxiety_model = None
suicide_model = None
feature_columns = None
symptom_mappings = None
metadata = None
anxiety_threshold = 0.5

def load_models():
    """Load all models and preprocessor files"""
    global anxiety_model, suicide_model, feature_columns, symptom_mappings, metadata, anxiety_threshold
    
    try:
        # Load models
        anxiety_model = joblib.load('./models/anxiety_model.pkl')
        suicide_model = joblib.load('./models/suicide_model.pkl')
        
        # Load feature columns
        feature_columns = joblib.load('./models/feature_columns.pkl')
        
        # Load symptom mappings
        symptom_mappings = joblib.load('./models/symptom_mappings.pkl')
        
        # Load metadata
        metadata = joblib.load('./models/metadata.pkl')
        anxiety_threshold = metadata.get('anxiety_threshold', 0.5)
        
        print("="*50)
        print("MODELS LOADED SUCCESSFULLY!")
        print("="*50)
        print(f"  Anxiety model: {metadata.get('anxiety_model', 'Unknown')}")
        print(f"  Suicide model: {metadata.get('suicide_model', 'Unknown')}")
        print(f"  Anxiety threshold: {anxiety_threshold}")
        print(f"  Features: {len(feature_columns)}")
        print("="*50)
        return True
        
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_input(form_data):
    """Convert HTML form data to model features"""
    # Initialize all features to 0
    patient_data = {col: 0 for col in feature_columns}
    
    # Set Age_num
    age_range = form_data.get('age', '35-40')
    try:
        age_parts = age_range.split('-')
        age_num = (int(age_parts[0]) + int(age_parts[1])) / 2
    except:
        age_num = metadata.get('age_median', 32.5)
    
    patient_data['Age_num'] = age_num
    patient_data['Hour'] = datetime.now().hour
    
    # Apply symptom mappings
    for symptom, mapping in symptom_mappings.items():
        value = form_data.get(symptom.lower(), 'No')
        if value in mapping:
            col_name = mapping[value]
            if col_name in patient_data:
                patient_data[col_name] = 1
    
    # Create DataFrame with correct column order
    df_input = pd.DataFrame([patient_data])
    return df_input[feature_columns]


@app.route('/')
def index():
    symptom_options = {
        'Sad': ['No', 'Sometimes', 'Yes'],
        'Irritable': ['No', 'Sometimes', 'Yes'],
        'Sleep': ['No', 'Yes', 'Two or more days a week'],
        'Concentration': ['No', 'Yes', 'Often'],
        'Appetite': ['No', 'Yes', 'Not at all'],
        'Guilt': ['No', 'Yes', 'Maybe'],
        'Bonding': ['No', 'Sometimes', 'Yes']
    }
    age_ranges = ['25-30', '30-35', '35-40', '40-45', '50+']
    
    return render_template('index.html', 
                         symptom_options=symptom_options,
                         age_ranges=age_ranges)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if anxiety_model is None:
            return jsonify({'success': False, 'error': 'Models not loaded'}), 500
        
        # Get form data
        form_data = {
            'age': request.form.get('age', '35-40'),
            'sad': request.form.get('sad', 'No'),
            'irritable': request.form.get('irritable', 'No'),
            'sleep': request.form.get('sleep', 'No'),
            'concentration': request.form.get('concentration', 'No'),
            'appetite': request.form.get('appetite', 'No'),
            'guilt': request.form.get('guilt', 'No'),
            'bonding': request.form.get('bonding', 'No')
        }
        
        # Preprocess input
        X_input = preprocess_input(form_data)
        
        # Predict Anxiety
        anxiety_proba = anxiety_model.predict_proba(X_input)[0]
        anxiety_pred = 1 if anxiety_proba[1] > anxiety_threshold else 0
        anxiety_label = "Anxiety" if anxiety_pred == 1 else "No Anxiety"
        
        # Predict Suicide
        suicide_pred = suicide_model.predict(X_input)[0]
        suicide_proba = suicide_model.predict_proba(X_input)[0]
        
        suicide_labels = ['No', 'Yes', 'Not interested to say']
        suicide_label = suicide_labels[suicide_pred]
        
        return jsonify({
            'success': True,
            'anxiety': {
                'prediction': anxiety_label,
                'probability': float(anxiety_proba[1]),
                'probability_no': float(anxiety_proba[0])
            },
            'suicide': {
                'prediction': suicide_label,
                'probabilities': {
                    'No': float(suicide_proba[0]),
                    'Yes': float(suicide_proba[1]),
                    'Not interested to say': float(suicide_proba[2]) if len(suicide_proba) > 2 else 0
                }
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint"""
    try:
        healthy_data = {col: 0 for col in feature_columns}
        healthy_data['Age_num'] = metadata.get('age_median', 32.5)
        healthy_data['Hour'] = 14
        df_test = pd.DataFrame([healthy_data])[feature_columns]
        
        anxiety_proba = anxiety_model.predict_proba(df_test)[0][1]
        
        return jsonify({
            'models_loaded': anxiety_model is not None,
            'anxiety_threshold': anxiety_threshold,
            'feature_count': len(feature_columns),
            'healthy_test_anxiety_probability': float(anxiety_proba)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if load_models():
        print("\n" + "="*50)
        print("STARTING FLASK SERVER")
        print("="*50)
        print("Open http://127.0.0.1:5000 in your browser")
        print("Press Ctrl+C to stop")
        print("="*50)
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        print("\n❌ Failed to load models.")
        print("Please ensure model files exist in './models/' directory")