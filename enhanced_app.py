#!/usr/bin/env python3
"""
Enhanced Zaura Health - Drug Interaction Prediction API
Professional web application with dosage-based safety analysis, user management, and scientist data contribution
Powered by Z Corp.®
"""

from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import itertools
import logging
from typing import List, Dict, Optional, Tuple
import os
import sys
import hashlib
import sqlite3
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.secret_key = 'zaura_health_secret_key_2025'  # Change in production

# Global variables for model components
model = None
preprocessor = None
device = None
model_info = None

# Enhanced dosage safety thresholds (mg per 24 hours)
DOSAGE_SAFETY_THRESHOLDS = {
    'aspirin': {'safe_max': 1000, 'warning_max': 2000, 'dangerous_max': 4000},
    'metformin': {'safe_max': 2000, 'warning_max': 2550, 'dangerous_max': 3000},
    'lisinopril': {'safe_max': 40, 'warning_max': 60, 'dangerous_max': 80},
    'ibuprofen': {'safe_max': 1200, 'warning_max': 2400, 'dangerous_max': 3200},
    'tylenol': {'safe_max': 3000, 'warning_max': 4000, 'dangerous_max': 6000},
    'atorvastatin': {'safe_max': 80, 'warning_max': 120, 'dangerous_max': 160},
    # Add more drugs as needed
}

class UserManager:
    """Manage user authentication and roles"""
    
    def __init__(self):
        self.init_database()
    
    def init_database(self):
        """Initialize user and contribution databases"""
        conn = sqlite3.connect('zaura_health.db')
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,  -- 'doctor' or 'scientist'
                full_name TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Drug combinations contributions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drug_contributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contributor_id INTEGER,
                drug_combination TEXT NOT NULL,
                safety_label TEXT NOT NULL,  -- 'safe' or 'unsafe'
                dosage_info TEXT,
                notes TEXT,
                confidence_level INTEGER,  -- 1-5 scale
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (contributor_id) REFERENCES users (id)
            )
        ''')
        
        # Create default users if they don't exist
        self.create_default_users(cursor)
        
        conn.commit()
        conn.close()
    
    def create_default_users(self, cursor):
        """Create default doctor and scientist accounts"""
        default_users = [
            {
                'username': 'dr_smith',
                'password': 'doctor123',
                'role': 'doctor',
                'full_name': 'Dr. John Smith',
                'email': 'dr.smith@zauraheath.com'
            },
            {
                'username': 'scientist_jane',
                'password': 'science123',
                'role': 'scientist',
                'full_name': 'Dr. Jane Wilson',
                'email': 'j.wilson@zauraheath.com'
            }
        ]
        
        for user_data in default_users:
            cursor.execute('SELECT id FROM users WHERE username = ?', (user_data['username'],))
            if not cursor.fetchone():
                password_hash = hashlib.sha256(user_data['password'].encode()).hexdigest()
                cursor.execute('''
                    INSERT INTO users (username, password_hash, role, full_name, email)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_data['username'], password_hash, user_data['role'], 
                      user_data['full_name'], user_data['email']))
    
    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user and return user info"""
        conn = sqlite3.connect('zaura_health.db')
        cursor = conn.cursor()
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute('''
            SELECT id, username, role, full_name, email 
            FROM users 
            WHERE username = ? AND password_hash = ?
        ''', (username, password_hash))
        
        user = cursor.fetchone()
        if user:
            # Update last login
            cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                         (datetime.now(), user[0]))
            conn.commit()
            
            return {
                'id': user[0],
                'username': user[1],
                'role': user[2],
                'full_name': user[3],
                'email': user[4]
            }
        
        conn.close()
        return None
    
    def add_contribution(self, contributor_id: int, drug_combination: str, 
                        safety_label: str, dosage_info: str = None, 
                        notes: str = None, confidence_level: int = 3):
        """Add a new drug combination contribution"""
        conn = sqlite3.connect('zaura_health.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO drug_contributions 
            (contributor_id, drug_combination, safety_label, dosage_info, notes, confidence_level)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (contributor_id, drug_combination, safety_label, dosage_info, notes, confidence_level))
        
        conn.commit()
        conn.close()
        return cursor.lastrowid

user_manager = UserManager()

class AdvancedDrugInteractionPreprocessor:
    """Advanced preprocessor for drug interaction data with enhanced feature engineering"""
    
    def __init__(self, max_drugs=10):
        self.max_drugs = max_drugs
        self.drug_encoder = None
        self.label_encoder = None
        self.scaler = None
        self.drug_vocab_size = 0
        self.feature_dim = 0
        
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        features = self._create_features(df)
        return features
    
    def _create_features(self, df):
        """Create comprehensive feature set for drug combinations"""
        # Drug encoding features
        drug_features = self._encode_drugs(df)
        
        # Numerical features
        numerical_features = self._create_numerical_features(df)
        
        # Combine features
        if numerical_features.shape[1] > 0:
            final_features = np.hstack([drug_features, numerical_features])
        else:
            final_features = drug_features
            
        return final_features.astype(np.float32)
    
    def _encode_drugs(self, df):
        """Encode drug information efficiently"""
        drug_columns = [f'drug{i}' for i in range(1, self.max_drugs + 1)]
        encoded_features = np.zeros((len(df), self.max_drugs), dtype=np.int32)
        
        for i, col in enumerate(drug_columns):
            if col in df.columns:
                # Handle missing values
                filled_col = df[col].fillna('MISSING')
                
                for j, drug in enumerate(filled_col):
                    try:
                        encoded_features[j, i] = self.drug_encoder.transform([drug])[0]
                    except ValueError:
                        # Unknown drug
                        encoded_features[j, i] = self.drug_encoder.transform(['UNKNOWN'])[0]
            else:
                # Column doesn't exist
                missing_code = self.drug_encoder.transform(['MISSING'])[0]
                encoded_features[:, i] = missing_code
        
        return encoded_features
    
    def _create_numerical_features(self, df):
        """Create numerical features efficiently with proper handling of mixed data types"""
        numerical_features = []
        
        # Dosage features - handle mixed numeric/text data
        if 'doses_per_24_hrs' in df.columns:
            doses_numeric = self._extract_numeric_doses(df['doses_per_24_hrs'])
            doses_scaled = self.scaler.transform(doses_numeric.reshape(-1, 1))
            numerical_features.append(doses_scaled.flatten())
        
        # Drug count features
        if 'total_drugs' in df.columns:
            numerical_features.append(df['total_drugs'].fillna(0).values)
        
        # Dosage availability
        if 'has_dosage_info' in df.columns:
            numerical_features.append(df['has_dosage_info'].fillna(0).values)
        
        return np.array(numerical_features).T if numerical_features else np.zeros((len(df), 0))
    
    def _extract_numeric_doses(self, doses_series):
        """Extract numeric values from mixed doses_per_24_hrs column"""
        def convert_to_numeric(value):
            if pd.isna(value):
                return 0.0
            
            # Convert to string for processing
            str_value = str(value).strip()
            
            # Try to convert directly to float first
            try:
                return float(str_value)
            except ValueError:
                pass
            
            # Handle common text cases
            if str_value.upper() in ['TAB', 'VIAL', 'CAP', 'SUPP', 'TUBE', 'BAG', 'SYR']:
                return 1.0  # Assume 1 unit per day
            elif str_value.upper() in ['ML', 'UDCUP']:
                return 1.0  # Assume 1 mL or 1 cup per day
            else:
                # Try to extract numbers from strings like "250mg" or "500"
                import re
                numbers = re.findall(r'\d+(?:\.\d+)?', str_value)
                if numbers:
                    return float(numbers[0])
                else:
                    return 0.0  # Default for unrecognizable formats
        
        # Apply conversion to all values
        numeric_doses = doses_series.apply(convert_to_numeric)
        return numeric_doses.values

class AdvancedDrugInteractionNet(nn.Module):
    """Advanced Neural Network for Drug Interaction Prediction with drug embeddings"""
    
    def __init__(self, input_size, drug_vocab_size, embedding_dim=64, 
                 hidden_sizes=[256, 128, 64], num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        self.max_drugs = 10
        self.embedding_dim = embedding_dim
        
        # Drug embedding layer for better drug representation
        self.drug_embedding = nn.Embedding(drug_vocab_size, embedding_dim, padding_idx=0)
        
        # Calculate input size for main network
        embedding_features = self.max_drugs * embedding_dim
        numerical_features = input_size - self.max_drugs
        total_input_size = embedding_features + numerical_features
        
        # Main neural network layers with batch normalization
        layers = []
        prev_size = total_input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        self.main_layers = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Split input into drug features and numerical features
        drug_features = x[:, :self.max_drugs].long()
        numerical_features = x[:, self.max_drugs:]
        
        # Get drug embeddings
        drug_embeddings = self.drug_embedding(drug_features)
        drug_features_flat = drug_embeddings.view(batch_size, -1)
        
        # Combine features
        combined_features = torch.cat([drug_features_flat, numerical_features], dim=1)
        
        # Pass through main network
        x = self.main_layers(combined_features)
        x = self.output(x)
        
        return x

def load_model():
    """Load the trained model and preprocessor"""
    global model, preprocessor, device, model_info
    
    try:
        # Check device availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model info
        with open('models/enhanced_model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        
        # Load preprocessor
        with open('models/enhanced_drug_interaction_preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Initialize model architecture
        model = AdvancedDrugInteractionNet(
            input_size=model_info['model_architecture']['input_size'],
            drug_vocab_size=model_info['model_architecture']['drug_vocab_size'],
            embedding_dim=model_info['model_architecture']['embedding_dim'],
            hidden_sizes=model_info['model_architecture']['hidden_sizes'],
            num_classes=model_info['model_architecture']['num_classes'],
            dropout_rate=model_info['model_architecture']['dropout_rate']
        ).to(device)
        
        # Load trained weights
        model.load_state_dict(torch.load('models/best_enhanced_drug_interaction_model.pth', map_location=device))
        model.eval()
        
        logger.info("✓ Model loaded successfully")
        logger.info(f"Model accuracy: {model_info['performance']['test_accuracy']:.4f}")
        logger.info(f"Drug vocabulary size: {preprocessor.drug_vocab_size}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def check_dosage_safety(drug_name: str, dosage: float) -> Tuple[str, str]:
    """
    Check if dosage is safe, warning, or dangerous for a specific drug
    Returns: (safety_level, message)
    """
    drug_lower = drug_name.lower()
    
    if drug_lower in DOSAGE_SAFETY_THRESHOLDS:
        thresholds = DOSAGE_SAFETY_THRESHOLDS[drug_lower]
        
        if dosage <= thresholds['safe_max']:
            return 'safe', f"Dosage within safe range (≤{thresholds['safe_max']}mg/24h)"
        elif dosage <= thresholds['warning_max']:
            return 'warning', f"High dosage - monitor closely ({thresholds['safe_max']}-{thresholds['warning_max']}mg/24h)"
        elif dosage <= thresholds['dangerous_max']:
            return 'dangerous', f"Dangerous dosage - reduce immediately ({thresholds['warning_max']}-{thresholds['dangerous_max']}mg/24h)"
        else:
            return 'critical', f"CRITICAL: Dosage exceeds maximum safe limit (>{thresholds['dangerous_max']}mg/24h)"
    
    return 'unknown', f"No dosage information available for {drug_name}"

def analyze_multi_drug_interactions(drugs: List[str], dosages: List[float] = None) -> Dict:
    """
    Enhanced multi-drug interaction analysis with dosage considerations
    """
    if len(drugs) < 2:
        return {"error": "At least 2 drugs are required"}
    
    if len(drugs) > 10:
        return {"error": "Maximum 10 drugs supported"}
    
    # Clean drugs
    cleaned_drugs = [drug.strip().title() for drug in drugs if drug.strip()]
    
    if dosages is None:
        dosages = [1.0] * len(cleaned_drugs)
    elif len(dosages) < len(cleaned_drugs):
        # Pad with default dosage
        dosages.extend([1.0] * (len(cleaned_drugs) - len(dosages)))
    
    # Individual drug dosage analysis
    dosage_analysis = {}
    overall_dosage_risk = 'safe'
    
    for i, (drug, dosage) in enumerate(zip(cleaned_drugs, dosages)):
        safety_level, message = check_dosage_safety(drug, dosage)
        dosage_analysis[drug] = {
            'dosage': dosage,
            'safety_level': safety_level,
            'message': message
        }
        
        # Update overall risk
        if safety_level in ['critical', 'dangerous']:
            overall_dosage_risk = 'dangerous'
        elif safety_level == 'warning' and overall_dosage_risk != 'dangerous':
            overall_dosage_risk = 'warning'
    
    # Generate all combinations for interaction analysis
    all_combinations = []
    combination_results = {}
    
    for r in range(2, len(cleaned_drugs) + 1):
        for combo in itertools.combinations(enumerate(cleaned_drugs), r):
            drug_combo = [drug for idx, drug in combo]
            dosage_combo = [dosages[idx] for idx, drug in combo]
            all_combinations.append((drug_combo, dosage_combo))
    
    # Predict each combination
    for drug_combo, dosage_combo in all_combinations:
        result = predict_single_combination(drug_combo, sum(dosage_combo) / len(dosage_combo))
        combo_key = " + ".join(drug_combo)
        
        if "error" not in result:
            # Apply dosage-based safety modification
            if overall_dosage_risk == 'dangerous':
                result['prediction'] = 'unsafe'
                result['confidence'] = min(result['confidence'], 0.8)
                result['dosage_override'] = True
                result['dosage_message'] = 'Marked unsafe due to dangerous dosage levels'
            elif overall_dosage_risk == 'warning':
                result['confidence'] = min(result['confidence'], 0.9)
                result['dosage_warning'] = True
                result['dosage_message'] = 'Monitor closely due to high dosages'
        
        combination_results[combo_key] = result
    
    # Overall safety assessment
    safe_combinations = [k for k, v in combination_results.items() if v.get('prediction') == 'safe']
    unsafe_combinations = [k for k, v in combination_results.items() if v.get('prediction') == 'unsafe']
    
    return {
        "status": "success",
        "input_drugs": cleaned_drugs,
        "dosages": dosages,
        "dosage_analysis": dosage_analysis,
        "overall_dosage_risk": overall_dosage_risk,
        "total_combinations": len(all_combinations),
        "safe_combinations": len(safe_combinations),
        "unsafe_combinations": len(unsafe_combinations),
        "overall_safety": "SAFE" if len(unsafe_combinations) == 0 and overall_dosage_risk != 'dangerous' else "UNSAFE",
        "safety_score": len(safe_combinations) / len(all_combinations) if all_combinations else 0,
        "detailed_results": combination_results,
        "safe_combination_list": safe_combinations,
        "unsafe_combination_list": unsafe_combinations
    }

def predict_single_combination(drug_combo: List[str], avg_dosage: float) -> Dict:
    """Predict safety for a single drug combination"""
    try:
        # Create prediction DataFrame
        prediction_data = {}
        
        # Fill drug columns
        for i in range(1, 11):
            col_name = f'drug{i}'
            if i <= len(drug_combo):
                prediction_data[col_name] = [drug_combo[i-1]]
            else:
                prediction_data[col_name] = [None]
        
        # Add other features
        prediction_data['doses_per_24_hrs'] = [avg_dosage]
        prediction_data['total_drugs'] = [len(drug_combo)]
        prediction_data['has_dosage_info'] = [1]
        prediction_data['subject_id'] = [0]
        prediction_data['drug_combination_id'] = ['_'.join(drug_combo)]
        
        df_pred = pd.DataFrame(prediction_data)
        
        # Transform using preprocessor
        X_pred = preprocessor.transform(df_pred)
        
        # Make prediction
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_pred).to(device)
            output = model(X_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0][pred_class].item()
        
        # Convert prediction to label
        prediction = preprocessor.label_encoder.inverse_transform([pred_class])[0]
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "safe_probability": probs[0][0].item(),
            "unsafe_probability": probs[0][1].item(),
            "drug_count": len(drug_combo)
        }
    except Exception as e:
        logger.error(f"Error predicting combination {drug_combo}: {e}")
        return {"error": str(e)}

# Authentication routes
@app.route('/login')
def login_page():
    """Serve login page"""
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    """Handle login requests"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400
        
        user = user_manager.authenticate(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['full_name'] = user['full_name']
            
            return jsonify({
                "success": True,
                "user": {
                    "username": user['username'],
                    "role": user['role'],
                    "full_name": user['full_name']
                },
                "redirect": "/"
            })
        else:
            return jsonify({"error": "Invalid credentials"}), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"error": "Login failed"}), 500

@app.route('/logout')
def logout():
    """Handle logout"""
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/')
def enhanced_dashboard():
    """Serve the enhanced main application page"""
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    
    return render_template('enhanced_index.html', user=session)

@app.route('/prescription-upload')
def prescription_upload():
    """Serve the prescription upload page"""
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    
    # Only allow doctors to access prescription upload
    if session.get('role') != 'doctor':
        flash('Access denied. Doctors only.', 'error')
        return redirect(url_for('enhanced_dashboard'))
    
    return render_template('prescription_upload.html', user=session)

@app.route('/contribute')
def contribute_page():
    """Serve the contribution page for scientists"""
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    
    if session.get('role') != 'scientist':
        flash('Access denied. Scientists only.', 'error')
        return redirect(url_for('index'))
    
    return render_template('contribute.html', user=session)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Enhanced API endpoint for drug interaction prediction"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        drugs = data.get('drugs', [])
        dosages = data.get('dosages', [])
        
        if not drugs:
            return jsonify({"error": "No drugs provided"}), 400
        
        # Validate dosages
        if dosages:
            try:
                dosages = [float(d) if d else 1.0 for d in dosages]
                for dosage in dosages:
                    if dosage < 0:
                        return jsonify({"error": "Dosages must be positive"}), 400
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid dosage format"}), 400
        
        # Make enhanced prediction
        result = analyze_multi_drug_interactions(drugs, dosages)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/contribute', methods=['POST'])
def api_contribute():
    """API endpoint for scientists to contribute new drug combinations"""
    try:
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        
        if session.get('role') != 'scientist':
            return jsonify({"error": "Access denied. Scientists only."}), 403
        
        data = request.get_json()
        
        drugs = data.get('drugs', [])
        safety_label = data.get('safety_label')
        dosage_info = data.get('dosage_info')
        notes = data.get('notes')
        confidence_level = data.get('confidence_level', 3)
        
        if not drugs or len(drugs) < 2:
            return jsonify({"error": "At least 2 drugs required"}), 400
        
        if safety_label not in ['safe', 'unsafe']:
            return jsonify({"error": "Safety label must be 'safe' or 'unsafe'"}), 400
        
        drug_combination = ' + '.join([drug.strip().title() for drug in drugs if drug.strip()])
        
        contribution_id = user_manager.add_contribution(
            session['user_id'], 
            drug_combination,
            safety_label,
            dosage_info,
            notes,
            confidence_level
        )
        
        return jsonify({
            "success": True,
            "contribution_id": contribution_id,
            "message": "Contribution added successfully"
        })
        
    except Exception as e:
        logger.error(f"Contribution error: {e}")
        return jsonify({"error": "Failed to add contribution"}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None,
        "accuracy": model_info['performance']['test_accuracy'] if model_info else None,
        "drug_vocabulary_size": preprocessor.drug_vocab_size if preprocessor else None,
        "authenticated": 'user_id' in session,
        "user_role": session.get('role', 'guest')
    })

@app.route('/api/drug-suggestions')
def drug_suggestions():
    """Get drug suggestions for autocomplete"""
    try:
        if 'user_id' not in session:
            return jsonify([])
        
        query = request.args.get('q', '').lower()
        
        if not query or len(query) < 1:
            return jsonify([])
        
        # Get common drug names from the preprocessor's vocabulary
        if preprocessor and hasattr(preprocessor, 'drug_encoder'):
            all_drugs = list(preprocessor.drug_encoder.classes_)
            
            # Enhanced matching: exact, starts with, contains
            suggestions = []
            query_lower = query.lower()
            
            # First: exact matches (case insensitive)
            exact_matches = [drug for drug in all_drugs 
                           if drug.lower() == query_lower 
                           and drug not in ['UNKNOWN', 'MISSING']]
            suggestions.extend(exact_matches)
            
            # Second: starts with query
            starts_with = [drug for drug in all_drugs 
                          if drug.lower().startswith(query_lower) 
                          and drug not in ['UNKNOWN', 'MISSING']
                          and drug not in exact_matches]
            suggestions.extend(starts_with[:10])
            
            # Third: contains query
            contains = [drug for drug in all_drugs 
                       if query_lower in drug.lower() 
                       and drug not in ['UNKNOWN', 'MISSING']
                       and drug not in exact_matches 
                       and drug not in starts_with]
            suggestions.extend(contains[:10])
            
            # Remove duplicates while preserving order
            seen = set()
            final_suggestions = []
            for drug in suggestions:
                if drug not in seen and len(drug) > 2:
                    seen.add(drug)
                    final_suggestions.append(drug)
            
            return jsonify(final_suggestions[:20])
        
        return jsonify([])
        
    except Exception as e:
        logger.error(f"Error getting drug suggestions: {e}")
        return jsonify([])

@app.route('/api/analyze_prescription', methods=['POST'])
def analyze_prescription():
    """Analyze uploaded prescription image for drug interactions"""
    try:
        # Check authentication
        if 'user_id' not in session:
            return jsonify({
                'success': False,
                'message': 'Authentication required'
            }), 401
        
        # Check doctor role
        if session.get('role') != 'doctor':
            return jsonify({
                'success': False,
                'message': 'Access denied. Doctors only.'
            }), 403
        
        # Check if file is present
        if 'prescription_image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No image file provided'
            }), 400
        
        file = request.files['prescription_image']
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
        
        # Get analysis options
        options_str = request.form.get('options', '{}')
        try:
            options = json.loads(options_str)
        except json.JSONDecodeError:
            options = {}
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_extension not in allowed_extensions:
            return jsonify({
                'success': False,
                'message': 'Invalid file type. Please upload an image or PDF file.'
            }), 400
        
        # TODO: Implement OCR processing when available
        # For now, return placeholder response indicating OCR is under development
        
        logger.info(f"Prescription upload attempt by user {session['user_id']} with file: {file.filename}")
        
        # Placeholder response - replace with actual OCR processing later
        placeholder_response = {
            'success': True,
            'message': 'OCR processing is currently under development',
            'data': {
                'ocr_status': 'development',
                'extracted_drugs': [
                    {'name': 'Aspirin', 'dosage': '81mg', 'confidence': 0.85},
                    {'name': 'Metformin', 'dosage': '500mg', 'confidence': 0.92},
                    {'name': 'Lisinopril', 'dosage': '10mg', 'confidence': 0.78}
                ],
                'interactions': {
                    'overall_risk': 'LOW',
                    'summary': 'No significant drug interactions detected based on extracted medications.',
                    'details': []
                },
                'recommendations': [
                    {
                        'type': 'info',
                        'title': 'OCR Development Status',
                        'text': 'Optical Character Recognition for handwritten prescriptions is currently being developed. This is placeholder data for demonstration purposes.'
                    },
                    {
                        'type': 'info',
                        'title': 'Monitor Blood Pressure',
                        'text': 'With Lisinopril therapy, regular blood pressure monitoring is recommended.'
                    },
                    {
                        'type': 'info',
                        'title': 'Take with Food',
                        'text': 'Metformin should be taken with meals to reduce gastrointestinal side effects.'
                    }
                ],
                'file_info': {
                    'filename': file.filename,
                    'size': len(file.read()),
                    'type': file.content_type
                }
            }
        }
        
        # Reset file pointer after reading for size
        file.seek(0)
        
        return jsonify(placeholder_response)
    
    except Exception as e:
        logger.error(f"Error analyzing prescription: {e}")
        return jsonify({
            'success': False,
            'message': f'Analysis failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Load model on startup
    logger.info("Loading Zaura Health Enhanced Drug Interaction Model...")
    
    if not load_model():
        logger.error("Failed to load model. Please ensure model files exist.")
        sys.exit(1)
    
    logger.info("✓ Zaura Health Enhanced API ready! (Powered by Z Corp.®)")
    logger.info("✓ User authentication system initialized")
    logger.info("✓ Dosage-based safety analysis enabled")
    logger.info("✓ Multi-drug interaction analysis ready")
    logger.info("Access the application at: http://localhost:5000")
    
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=True)