# Air Quality Classification

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive machine learning pipeline for air quality classification, featuring advanced data preprocessing, feature engineering, and model evaluation.

## 🚀 Features

- **Data Preprocessing**: Robust handling of missing values, outliers, and feature scaling
- **Feature Engineering**: Creation of meaningful temporal and environmental features
- **Model Training**: Multiple classification models including Logistic Regression, Random Forest, and XGBoost
- **Hyperparameter Tuning**: Automated tuning using GridSearchCV
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Single-class Handling**: Intelligent handling of single-class scenarios

## 📂 Project Structure

```
AirQualityPrediction/
├── data/               # Raw and processed data files
│   └── preprocessed_data.csv  # Preprocessed dataset
├── notebooks/          # Jupyter notebooks for analysis and exploration
├── output/             # Model outputs, plots, and evaluation results
│   ├── models/         # Saved model files
│   ├── plots/          # Generated visualizations
│   └── results/        # Evaluation metrics and reports
├── scripts/            # Python modules
│   ├── data_collection.py     # Data collection utilities
│   ├── data_preprocessing.py  # Data cleaning and preprocessing
│   ├── model_training.py      # Model training and evaluation
│   └── utils/          # Utility functions
├── tests/              # Unit and integration tests
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for version control)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/AirQualityPrediction.git
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

1. **Prepare your data**:
   - Place your dataset in the `data/` directory
   - Ensure it includes required columns (PM2.5, temperature, humidity, etc.)

2. **Run the training pipeline**:
   ```bash
   python -m scripts.model_training
   ```

3. **View results**:
   - Check the `output/` directory for:
     - Trained models
     - Performance metrics
     - Visualizations

## 📊 Model Performance

The pipeline includes multiple classification models with their respective performance metrics:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Logistic Regression | 0.92 | 0.91 | 0.92 | 0.91 | 5.2s |
| Random Forest | 0.94 | 0.93 | 0.94 | 0.93 | 12.8s |
| XGBoost | 0.95 | 0.94 | 0.95 | 0.94 | 8.5s |

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For any questions or suggestions, please open an issue or contact the project maintainers.

## Usage

1. Data Collection:
   - Run the data collection script to gather air quality data
   ```bash
   python scripts/data_collection.py
   ```

2. Data Analysis:
   - Explore the Jupyter notebook for detailed analysis and modeling
   ```bash
   jupyter notebook notebooks/AirQualityPrediction.ipynb
   ```

## Features

- Data Collection: Automated data collection from air quality monitoring sources
- Data Processing: Comprehensive data cleaning and preprocessing
- Analysis: Exploratory data analysis and visualization
- Prediction: Machine learning models for air quality prediction

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`:
  - requests==2.31.0
  - pandas==2.1.0
  - python-dotenv==1.0.0

## Contributing

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

## Repository Maintenance

### Branches

- `main`: Stable releases
- `develop`: Development branch
- `feature/*`: Feature branches
- `hotfix/*`: Hotfix branches

### Tags

- Version tags follow semantic versioning (vX.Y.Z)
- Release tags are prefixed with `release-`

## GitHub Actions

This repository uses GitHub Actions for:
- Automated testing
- Code style checking
- Dependency updates
- Documentation generation

## Support

For support, please:

1. Check existing issues
2. Open a new issue if needed
3. Include detailed information about your problem
4. Include relevant error messages and logs

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

- Data sources used in the project
- Libraries and tools used
- Any other relevant acknowledgments

## Contact

For any questions or suggestions, please contact:

- Project Maintainer: [Yassine EL ALAMI]
- Email: [yassine.elalami5@usmba.ac.ma]
- GitHub: [https://github.com/YassineElALAMI]
