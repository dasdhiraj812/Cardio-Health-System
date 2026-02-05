# Cardiovascular Disease Prediction System

A complete AI/ML-powered web application for predicting cardiovascular disease risk using machine learning.

## 📋 Project Overview

This project combines machine learning with a modern web interface to provide personalized cardiovascular disease risk assessments. It uses a Random Forest classifier trained on 70,000+ patient records to predict disease probability based on health metrics.

## 🎯 Features

- **AI-Powered Predictions**: Random Forest ML model trained on comprehensive patient data
- **Real-time Risk Assessment**: Instant predictions with probability scores
- **Interactive Dashboard**: Beautiful, responsive web interface
- **Data Analytics**: Comprehensive statistics and visualizations
- **Multi-page Application**: Home, Prediction, Analytics, and About pages
- **RESTful API**: Backend API for all predictions and data
- **Mobile Responsive**: Works seamlessly on all devices

## 🛠️ Technology Stack

### Backend
- **Framework**: Flask (Python)
- **ML Library**: scikit-learn
- **Data Processing**: pandas, numpy
- **API**: RESTful with CORS support

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Responsive styling
- **JavaScript** - Interactive functionality
- **Chart.js** - Data visualization

### Machine Learning
- **Algorithm**: Random Forest Classifier
- **Estimators**: 100 decision trees
- **Features**: 11 health metrics
- **Training Data**: 70,000+ patient records

## 📁 Project Structure

```
MAJOR PROJECT 4/
├── app.py                      # Flask backend application
├── train_model.py              # Model training script
├── cardio_train (1).csv        # Dataset
├── requirements.txt            # Python dependencies
├── templates/                  # HTML templates
│   ├── index.html             # Home page
│   ├── predict.html           # Prediction page
│   ├── analytics.html         # Analytics page
│   └── about.html             # About page
├── static/                     # Static files
│   ├── style.css              # Global styles
│   ├── script.js              # Global scripts
│   ├── predict.js             # Prediction page logic
│   └── analytics.js           # Analytics page logic
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip (Python package manager)
- Modern web browser

### Installation

1. **Clone or navigate to the project directory**
```bash
cd "MAJOR PROJECT 4"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the machine learning model**
```bash
python train_model.py
```

This will:
- Load the cardiovascular disease dataset
- Preprocess the data
- Train a Random Forest classifier
- Save the model, scaler, and feature names

**Expected output:**
```
✓ Model trained and saved successfully!
Files saved: cardio_model.pkl, scaler.pkl, feature_names.pkl
```

4. **Start the Flask server**
```bash
python app.py
```

The server will start at `http://localhost:5000`

## 📖 Usage

### Home Page (/)
- Overview of the application
- Key features highlight
- Navigation to other pages

### Prediction Page (/predict)
1. Enter your health information:
   - Age (in days)
   - Gender
   - Height & Weight
   - Blood Pressure (Systolic & Diastolic)
   - Cholesterol & Glucose Levels
   - Lifestyle factors (Smoking, Alcohol, Activity)

2. Click "Get Prediction"

3. Receive personalized assessment:
   - Disease probability percentage
   - Risk level (Low/Moderate/High)
   - Visual risk meter
   - Personalized recommendations

### Analytics Page (/analytics)
- Total patient records in dataset
- Disease vs healthy case distribution
- Age distribution analysis
- Feature statistics (min, max, average)
- Key insights about the data

### About Page (/about)
- Detailed project information
- Technology stack details
- Input parameter descriptions
- How predictions work
- Risk assessment levels
- Medical disclaimer
- Dataset information

## 📊 Input Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| Age | Patient age in days | Variable |
| Gender | 1=Male, 2=Female | 1-2 |
| Height | Height in centimeters | cm |
| Weight | Weight in kilograms | kg |
| Systolic BP | Systolic blood pressure | mmHg |
| Diastolic BP | Diastolic blood pressure | mmHg |
| Cholesterol | Level (0=Normal to 3=High) | 0-3 |
| Glucose | Level (0=Normal to 3=High) | 0-3 |
| Smoking | Current smoker? | 0 or 1 |
| Alcohol | Alcohol consumption? | 0 or 1 |
| Activity | Physically active? | 0 or 1 |

## 🔮 Risk Assessment Levels

- **🟢 Low Risk (0-30%)**: Maintain healthy lifestyle habits
- **🟡 Moderate Risk (30-60%)**: Consider consulting healthcare provider
- **🔴 High Risk (60%+)**: Please consult healthcare professional

## 🔗 API Endpoints

### POST /api/predict
Make a single prediction
```json
{
    "age": 18393,
    "gender": 2,
    "height": 168,
    "weight": 62.0,
    "ap_hi": 110,
    "ap_lo": 80,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1
}
```

### GET /api/statistics
Get dataset statistics

### GET /api/model-info
Get model information

### GET /api/health
Health check endpoint

### POST /api/batch-predict
Batch prediction for multiple records

## 📈 Model Performance

- **Algorithm**: Random Forest Classifier
- **Training Data**: 70,000+ records
- **Features**: 11 health-related metrics
- **Estimators**: 100 decision trees
- **Max Depth**: 20 levels

## ⚠️ Important Disclaimer

**Medical Disclaimer**: This tool is for educational and informational purposes only and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns. The predictions made by this system are based on machine learning models and may not be 100% accurate.

## 🔒 Privacy & Security

- Data is processed in real-time and not stored
- No user information is logged or saved
- All computations happen locally on the server
- CORS enabled for secure cross-origin requests

## 📝 File Descriptions

### train_model.py
- Loads and preprocesses cardiovascular disease data
- Trains Random Forest classifier
- Evaluates model performance
- Saves trained model and scaler

### app.py
- Flask application with all routes
- RESTful API endpoints
- Frontend route handlers
- Model prediction logic

### Templates
- **index.html**: Home page with hero section and features
- **predict.html**: Interactive prediction form and results display
- **analytics.html**: Dataset statistics and visualizations
- **about.html**: Detailed project and model information

### Static Files
- **style.css**: Complete styling for all pages
- **script.js**: Shared utility functions and API helpers
- **predict.js**: Prediction form handling and results display
- **analytics.js**: Analytics page data loading and charting

## 🐛 Troubleshooting

### Model not found error
- Run `python train_model.py` to train the model first
- Ensure model files are in the project root directory

### Port already in use
- Change port in app.py: `app.run(port=5001)`

### CORS errors
- Ensure Flask-CORS is installed: `pip install Flask-CORS`

### API not responding
- Verify Flask server is running
- Check browser console for errors
- Ensure API endpoints are correctly referenced

## 📚 Learning Resources

- **Scikit-learn**: https://scikit-learn.org/
- **Flask**: https://flask.palletsprojects.com/
- **Chart.js**: https://www.chartjs.org/
- **Machine Learning**: https://www.coursera.org/learn/machine-learning

## 🎓 Educational Purpose

This project is designed as an educational tool to demonstrate:
- Machine learning model development
- Flask web framework usage
- RESTful API design
- Frontend-backend integration
- Data visualization
- Responsive web design

## 👨‍💻 Development

### Adding New Features
1. Update backend routes in `app.py`
2. Create corresponding frontend in templates
3. Add styling to `style.css`
4. Add JavaScript functionality to static files

### Improving Model
1. Experiment with different algorithms
2. Tune hyperparameters
3. Add feature engineering
4. Evaluate on validation set

## 📞 Support

For questions or issues:
1. Check the About page for detailed documentation
2. Review API documentation in app.py
3. Check browser console for JavaScript errors
4. Verify Python package versions in requirements.txt

## 📄 License

This project is for educational purposes. Please respect copyright and privacy regulations when using medical data.

## 🙏 Acknowledgments

- Dataset sourced from cardiovascular disease research
- Built with Flask, scikit-learn, and modern web technologies
- Inspired by machine learning best practices

---

**Version**: 1.0.0  
**Last Updated**: February 2024  
**Status**: Production Ready
#   C a r d i o - H e a l t h - S y s t e m  
 