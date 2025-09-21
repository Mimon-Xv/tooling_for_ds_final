"""
Configuration for realistic feature ranges and descriptions.
This helps make the diabetes prediction interface more user-friendly.
"""

# Realistic ranges for diabetes features (based on medical literature)
REALISTIC_RANGES = {
    'age': {
        'min': 20,
        'max': 80,
        'unit': 'years',
        'description': 'Patient age in years'
    },
    'sex': {
        'min': 0,
        'max': 1,
        'unit': 'binary',
        'description': 'Sex (0 = female, 1 = male)'
    },
    'bmi': {
        'min': 15.0,
        'max': 50.0,
        'unit': 'kg/m²',
        'description': 'Body Mass Index'
    },
    'bp': {
        'min': 80,
        'max': 200,
        'unit': 'mmHg',
        'description': 'Average blood pressure'
    },
    's1': {
        'min': 50,
        'max': 300,
        'unit': 'mg/dl',
        'description': 'Total cholesterol'
    },
    's2': {
        'min': 50,
        'max': 300,
        'unit': 'mg/dl',
        'description': 'Low-density lipoproteins'
    },
    's3': {
        'min': 20,
        'max': 100,
        'unit': 'mg/dl',
        'description': 'High-density lipoproteins'
    },
    's4': {
        'min': 5,
        'max': 50,
        'unit': 'mg/dl',
        'description': 'Total cholesterol / HDL'
    },
    's5': {
        'min': 3.0,
        'max': 8.0,
        'unit': 'log(mg/dl)',
        'description': 'Log of serum triglycerides'
    },
    's6': {
        'min': 70,
        'max': 200,
        'unit': 'mg/dl',
        'description': 'Blood sugar level'
    }
}

# Feature names with better descriptions
FEATURE_DESCRIPTIONS = {
    'age': 'Age',
    'sex': 'Sex',
    'bmi': 'BMI',
    'bp': 'Blood Pressure',
    's1': 'Total Cholesterol',
    's2': 'LDL Cholesterol',
    's3': 'HDL Cholesterol',
    's4': 'Cholesterol Ratio',
    's5': 'Log Triglycerides',
    's6': 'Blood Sugar'
}

def get_realistic_range(feature_name):
    """Get realistic range for a feature."""
    return REALISTIC_RANGES.get(feature_name, {
        'min': -1.0,
        'max': 1.0,
        'unit': 'normalized',
        'description': 'Normalized value'
    })

def get_feature_description(feature_name):
    """Get human-readable description for a feature."""
    return FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
