# Zaura Health - Drug Interaction Analysis Platform

## 🏥 Professional AI-Powered Drug Safety Analysis

Zaura Health is a cutting-edge web application that uses advanced deep learning to analyze drug interactions and ensure patient safety. Built with state-of-the-art neural networks and a professional interface.

![Zaura Health Banner](https://img.shields.io/badge/Zaura-Health-blue?style=for-the-badge&logo=medical-cross)p.® - Drug Interaction Analysis Platform

## � Professional AI-Powered Drug Safety Analysis

Z Corp.® is a cutting-edge web application that uses advanced deep learning to analyze drug interactions and ensure patient safety. Built with state-of-the-art neural networks and a professional interface.

![Zaura Healthcare Banner](https://img.shields.io/badge/Zaura%E2%84%A2-Healthcare-blue?style=for-the-badge&logo=medical-cross)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-2.3+-green?style=flat-square&logo=flask)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-87.5%25-brightgreen?style=flat-square)

## ✨ Key Features

### 🧠 Advanced AI Technology
- **Deep Learning Model**: 910K+ parameter neural network with drug embeddings
- **High Accuracy**: 87.52% test accuracy on 20M+ interaction records
- **CUDA Acceleration**: GPU-optimized for fast predictions
- **Multi-drug Support**: Analyze up to 10 drugs simultaneously

### 🎨 Professional Interface
- **Healthcare Branding**: Professional Zaura™ design system
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Validation**: Instant feedback on drug entries
- **Interactive Results**: Comprehensive safety analysis with confidence scores

### 🔬 Comprehensive Analysis
- **Parallel Processing**: All drug combinations analyzed simultaneously  
- **Confidence Scoring**: AI certainty levels for each prediction
- **Safety Recommendations**: Professional medical guidance
- **Detailed Reporting**: Individual combination breakdowns

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 8GB RAM recommended (4GB minimum)
- CUDA-compatible GPU (optional, for acceleration)

### Windows Setup

1. **Clone and Navigate**
   ```cmd
   cd "c:\Users\tendo\Documents\Srivatsav\sem 5\HPC"
   ```

2. **Run Setup Script**
   ```cmd
   setup.bat
   ```

3. **Start Application** 
   ```cmd
   run.bat
   ```

4. **Access Web Interface**
   Open your browser to: `http://localhost:5000`

### Manual Setup (Alternative)

1. **Create Virtual Environment**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```cmd
   pip install -r requirements.txt
   ```

3. **Ensure Model Files Exist**
   - `best_enhanced_drug_interaction_model.pth`
   - `enhanced_drug_interaction_preprocessor.pkl` 
   - `enhanced_model_info.pkl`

4. **Start Server**
   ```cmd
   python app.py
   ```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 87.52% |
| **Model Parameters** | 910,274 |
| **Training Dataset** | 20M+ records |
| **Drug Vocabulary** | 1000+ unique drugs |
| **Supported Combinations** | 2-10 drugs per analysis |

## 🏗️ Architecture

### Backend Components
- **Flask API Server**: RESTful endpoints for predictions
- **PyTorch Model**: Advanced neural network with embeddings
- **Data Preprocessing**: Intelligent drug name handling
- **Error Handling**: Comprehensive validation and error reporting

### Frontend Components
- **HTML5**: Semantic, accessible markup
- **CSS3**: Professional healthcare styling with animations
- **JavaScript (ES6+)**: Interactive drug management and API integration
- **Responsive Design**: Mobile-first approach

### Model Architecture
```
Input Layer (Drug Embeddings + Numerical Features)
    ↓
Embedding Layer (64-dim drug representations)
    ↓
Hidden Layer 1 (256 neurons + BatchNorm + ReLU + Dropout)
    ↓
Hidden Layer 2 (128 neurons + BatchNorm + ReLU + Dropout)
    ↓  
Hidden Layer 3 (64 neurons + BatchNorm + ReLU + Dropout)
    ↓
Output Layer (2 classes: Safe/Unsafe)
```

## 🎯 Usage Guide

### Basic Analysis
1. **Enter Drugs**: Add 2-10 drug names using the form
2. **Optional Dosage**: Include doses per 24 hours if known
3. **Analyze**: Click "Analyze Drug Interactions" 
4. **Review Results**: Get comprehensive safety assessment

### Understanding Results
- **SAFE**: No dangerous interactions detected
- **UNSAFE**: Potential dangerous interactions found
- **Confidence**: Model certainty (0-100%, higher is better)
- **Safety Score**: Overall combination safety percentage

### Professional Features
- **Keyboard Shortcuts**: Ctrl+Enter to analyze, Ctrl+Plus to add drugs
- **Real-time Validation**: Instant feedback on drug name formatting
- **Error Handling**: Clear error messages with suggested fixes
- **Accessibility**: Screen reader compatible, keyboard navigation

## 🔒 Safety & Compliance

### Important Medical Disclaimer
> **⚠️ MEDICAL DISCLAIMER**: This tool provides AI-based analysis for **informational purposes only**. Always consult qualified healthcare professionals for medical decisions. This system should not replace professional medical advice, diagnosis, or treatment.

### Data Privacy
- No personal health information stored
- Drug queries processed in memory only
- No data transmission to external servers
- HIPAA-conscious design principles

## 📁 Project Structure

```
zaura-healthcare/
├── app.py                          # Flask application server
├── requirements.txt                # Python dependencies
├── setup.bat                       # Windows setup script
├── run.bat                         # Application launcher
├── README.md                       # This documentation
├── templates/
│   └── index.html                  # Main web interface
├── static/
│   ├── css/
│   │   └── style.css              # Professional styling
│   ├── js/
│   │   └── app.js                 # Interactive functionality
│   └── images/                     # Brand assets
├── best_enhanced_drug_interaction_model.pth     # Trained model weights
├── enhanced_drug_interaction_preprocessor.pkl   # Data preprocessor
└── enhanced_model_info.pkl                      # Model metadata
```

## 🛠️ API Endpoints

### `POST /api/predict`
Analyze drug interactions for given combinations.

**Request Body:**
```json
{
    "drugs": ["Aspirin", "Warfarin", "Metformin"],
    "dosage": 2.0
}
```

**Response:**
```json
{
    "status": "success",
    "overall_safety": "UNSAFE", 
    "safety_score": 0.667,
    "total_combinations": 6,
    "safe_combinations": 4,
    "unsafe_combinations": 2,
    "detailed_results": { ... }
}
```

### `GET /api/health`
Check API and model status.

### `GET /api/drug-suggestions?q=aspir`
Get drug name suggestions for autocomplete.

## 🧪 Development

### Running Tests
```cmd
python -m pytest tests/
```

### Model Retraining
1. Update dataset in Jupyter notebook
2. Run all cells to retrain model
3. Ensure model files are saved to project directory
4. Restart web application

### Custom Styling
Edit `static/css/style.css` to customize the interface while maintaining healthcare design principles.

## 📈 Performance Optimization

### Production Deployment
- Use `gunicorn` or `waitress` for production serving
- Enable CUDA acceleration with compatible GPU
- Implement Redis caching for frequent queries
- Set up load balancing for high traffic

### Model Optimization
- Batch prediction requests when possible
- Use model quantization for faster inference
- Implement result caching for common combinations

## 🏥 Brand Guidelines

### Zaura Healthcare™ Brand Elements
- **Primary Colors**: Professional blue (#2563eb) and healthcare green (#10b981)
- **Typography**: Inter font family for modern readability
- **Logo**: Heartbeat icon with gradient styling
- **Trademark**: Always display "Zaura Healthcare™" with trademark symbol

## 🤝 Contributing

### Code Standards
- Follow PEP 8 for Python code
- Use semantic HTML5 elements
- Maintain WCAG 2.1 accessibility standards
- Include comprehensive error handling

### Medical Accuracy
- Validate against established drug interaction databases
- Include confidence intervals in predictions
- Provide clear uncertainty indicators
- Maintain conservative safety recommendations

## 📞 Support

For technical support or medical guidance questions:
- **Technical Issues**: Check console logs and model file integrity
- **Model Questions**: Verify training data and preprocessing steps
- **Medical Guidance**: Always consult qualified healthcare professionals

## 📄 License

**Zaura Healthcare™** - Professional Drug Interaction Analysis Platform
© 2025 Zaura™ Healthcare. All rights reserved.

*This software is for educational and informational purposes only. Not intended for clinical decision-making without professional medical oversight.*

---

### 🔬 Built with Advanced Technology
- **Deep Learning**: PyTorch neural networks with embeddings
- **Web Framework**: Flask with CORS support
- **Frontend**: Modern ES6+ JavaScript with CSS3 animations  
- **Data Processing**: Pandas and NumPy for efficient computation
- **GPU Acceleration**: CUDA support for faster predictions

**Powered by Z Corp.® AI Technology**