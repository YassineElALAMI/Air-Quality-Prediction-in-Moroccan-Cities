# Air Quality Prediction in Moroccan Cities

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive machine learning pipeline for air quality prediction and classification in Moroccan cities, featuring advanced data preprocessing, feature engineering, and model evaluation.

## 🚀 Features

- **Data Collection**: Automated data gathering from air quality monitoring sources
- **Data Preprocessing**: Robust handling of missing values, outliers, and feature scaling
- **Feature Engineering**: Creation of meaningful temporal and environmental features
- **Model Training**: Multiple models including:
  - Regression models for air quality index prediction
  - Classification models for air quality categories
- **Hyperparameter Tuning**: Automated tuning using GridSearchCV
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Federated Learning**: Support for distributed model training across multiple locations

## 📂 Project Structure

```
AirQualityPrediction/
├── data/                           # Raw and processed data files
│   └── preprocessed_data.csv       # Preprocessed dataset
├── notebooks/                      # Jupyter notebooks for analysis
├── output/                         # Model outputs and results
│   ├── models/                     # Saved model files
│   ├── plots/                      # Generated visualizations
│   └── results/                    # Evaluation metrics and reports
├── scripts/                        # Python modules
│   ├── data_collection.py          # Data collection utilities
│   ├── data_preprocessing.py       # Data cleaning and preprocessing
│   ├── model_training_regression.py # Regression model training
│   ├── model_training_classification.py # Classification model training
│   └── federated_learning.py       # Federated learning implementation
├── requirements.txt                # Project dependencies
├── check_missingno.py              # Data quality checking
└── README.md                       # Project documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YassineElALAMI/Air-Quality-Prediction-in-Moroccan-Cities.git
   cd AirQualityPrediction
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Usage

### Data Collection

```bash
python scripts/data_collection.py
```

### Data Preprocessing

```bash
python scripts/data_preprocessing.py
```

### Model Training

For regression models:
```bash
python scripts/model_training_regression.py
```

For classification models:
```bash
python scripts/model_training_classification.py
```

### Federated Learning

```bash
python scripts/federated_learning.py
```

### Data Analysis

Explore the Jupyter notebooks for detailed analysis and visualization:
```bash
jupyter notebook notebooks/
```

## 📊 Model Performance

### Classification Models
| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Logistic Regression | 0.92 | 0.91 | 0.92 | 0.91 | 5.2s |
| Random Forest | 0.94 | 0.93 | 0.94 | 0.93 | 12.8s |
| XGBoost | 0.95 | 0.94 | 0.95 | 0.94 | 8.5s |

### Regression Models
| Model | MAE | RMSE | R² Score | Training Time |
|-------|-----|------|----------|---------------|
| Linear Regression | 2.1 | 3.5 | 0.89 | 1.2s |
| Random Forest | 1.8 | 2.9 | 0.92 | 15.3s |
| XGBoost | 1.7 | 2.7 | 0.93 | 9.8s |

## 🤝 Contributing

We welcome contributions from the community! Please follow these guidelines:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some Feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

### Pull Request Guidelines
- Ensure your code follows PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Reference any relevant issues in your PR description

## 📝 License

This project is licensed under the MIT License — see the LICENSE file.

## Contact

For any questions or suggestions, please contact:

- Project Maintainer: Yassine EL ALAMI
- Email: yassine.elalami5@usmba.ac.ma
- GitHub: https://github.com/YassineElALAM
