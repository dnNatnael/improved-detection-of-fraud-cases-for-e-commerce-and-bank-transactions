# Fraud Detection for E-commerce and Bank Transactions

![Python Version](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

> **Advanced fraud detection system for e-commerce transactions and credit card payments using machine learning**

## 📋 Table of Contents

- [Business Objective](#business-objective)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Usage Examples](#usage-examples)
- [Workflow](#workflow)
- [Results Organization](#results-organization)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

---

## 🎯 Business Objective

**Company:** Adey Innovations Inc.

This project aims to build a robust, production-ready fraud detection system capable of identifying fraudulent transactions across two domains:

1. **E-commerce Transactions**: Detect suspicious purchases based on user behavior, geolocation, and transaction patterns
2. **Credit Card Transactions**: Identify fraudulent bank transactions using advanced feature engineering and anomaly detection

### Goals
- ✅ Achieve high precision and recall in fraud detection
- ✅ Minimize false positives to avoid customer friction
- ✅ Handle severe class imbalance effectively
- ✅ Provide interpretable results for business stakeholders
- ✅ Build scalable, production-grade code

---

## 📊 Datasets

### 1. **Fraud_Data.csv** (E-commerce Transactions)
- **Size**: ~151,000 transactions
- **Features**: 
  - User information (ID, age, sex)
  - Transaction details (purchase value, time, signup time)
  - Device & browser information
  - Traffic source (SEO, Ads, Direct)
  - IP address for geolocation
- **Target**: `class` (0 = Legitimate, 1 = Fraud)
- **Class Distribution**: ~9% fraudulent transactions

### 2. **creditcard.csv** (Bank Credit Card Transactions)
- **Size**: ~284,000 transactions
- **Features**: 
  - 28 anonymized features (V1-V28) from PCA transformation
  - Transaction amount
  - Time elapsed from first transaction
- **Target**: `Class` (0 = Legitimate, 1 = Fraud)
- **Class Distribution**: Highly imbalanced (~0.17% fraud)

### 3. **IpAddress_to_Country.csv** (Geolocation Mapping)
- **Purpose**: Map IP addresses to countries for geographical fraud analysis
- **Features**: IP ranges and country codes

---

## 📁 Project Structure

```
improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions/
│
├── .github/                    # GitHub workflows and configurations
├── .vscode/                    # VS Code settings
├── data/
│   ├── raw/                   # Original datasets (DO NOT MODIFY)
│   │   ├── Fraud_Data.csv
│   │   ├── creditcard.csv
│   │   └── IpAddress_to_Country.csv
│   └── processed/             # Cleaned and preprocessed data
│       ├── fraud_data_cleaned.csv
│       ├── creditcard_cleaned.csv
│       ├── fraud_X_train_smote.csv
│       ├── fraud_y_train_smote.csv
│       ├── fraud_X_test.csv
│       ├── fraud_y_test.csv
│       ├── creditcard_X_train_smote.csv
│       ├── creditcard_y_train_smote.csv
│       ├── creditcard_X_test.csv
│       └── creditcard_y_test.csv
│
├── notebooks/                 # Jupyter notebooks for exploration
│   └── eda-fraud-data.ipynb  # Main EDA and preprocessing notebook
│
├── scripts/                   # Python scripts for automation
│   ├── train_fraud_models.py # Model training and evaluation pipeline
│   └── generate_report.py    # Report generation utilities
│
├── src/                       # Reusable Python modules
│   ├── data_loading.py       # Data loading with error handling
│   ├── data_cleaning.py      # Data cleaning and standardization
│   ├── feature_engineering.py # Feature engineering and geolocation
│   ├── eda_utils.py          # EDA utilities and visualizations
│   └── preprocessing.py      # Preprocessing pipeline (scaling, encoding, SMOTE)
│
├── tests/                     # Unit tests
│   ├── test_project_structure.py
│   └── __init__.py
│
├── models/                    # Saved trained models
│   └── *.pkl, *.joblib       # Trained model files
│
├── reports/                   # Model evaluation reports and analysis
│   └── (analysis documents and summaries)
│
├── temp_charts/              # Generated visualizations
│   └── *.png                 # Charts, plots, and graphs
│
├── requirements.txt          # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- Git (for version control)

### 1. Clone the Repository
```bash
git clone https://github.com/dnNatnael/improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions.git
cd improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning algorithms
- `imbalanced-learn` - SMOTE and class balancing
- `xgboost` - Gradient boosting ensemble models
- `joblib` - Model serialization
- `jupyter` - Interactive notebooks

### 4. Verify Installation
```bash
python -c "import pandas, numpy, sklearn, imblearn; print('✓ All packages installed successfully')"
```

---

## 🏃 How to Run

### Option 1: Run EDA Notebook (Recommended for Exploration)

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Navigate to**
   ```
   notebooks/eda-fraud-data.ipynb
   ```

3. **Run All Cells**
   - Click `Kernel` → `Restart & Run All`
   - Or run cells sequentially with `Shift + Enter`

### Option 2: Train Fraud Detection Models

**Train models on both datasets:**
```bash
python scripts/train_fraud_models.py
```

This script will:
- Load both `creditcard.csv` and `Fraud_Data.csv`
- Perform stratified train-test split
- Train Logistic Regression (baseline) and XGBoost (ensemble) models
- Perform 5-fold stratified cross-validation
- Compare models and select the best one
- Generate visualizations and save trained models

**Output locations:**
- Trained models: `models/`
- Visualizations: `temp_charts/`
- Console: Detailed metrics and recommendations

### Option 3: Use Individual Components

**Load and explore data:**
```python
from src.data_loading import load_data

# Load a dataset
df = load_data('data/raw/creditcard.csv')
print(df.head())
```

**Preprocess data:**
```python
from src.preprocessing import train_test_split_data, scale_features

# Split data
X_train, X_test, y_train, y_test = train_test_split_data(
    df, target_col='Class', test_size=0.2
)

# Scale features
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
```

**Train a custom model:**
```python
from scripts.train_fraud_models import FraudDetectionModelTrainer

trainer = FraudDetectionModelTrainer(
    dataset_path='data/raw/creditcard.csv',
    target_col='Class',
    dataset_name='Credit Card'
)
trainer.stratified_train_test_split()
trainer.train_baseline_model()
```

---

## 💻 Usage Examples

### Example 1: Complete Model Training Pipeline

Train models on both datasets with full evaluation:

```bash
# Activate virtual environment (if using one)
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Run the complete training pipeline
python scripts/train_fraud_models.py
```

**What happens:**
- Loads `data/raw/creditcard.csv` and `data/raw/Fraud_Data.csv`
- Performs stratified train-test split (80-20)
- Trains Logistic Regression (baseline) and XGBoost (ensemble) models
- Runs 5-fold stratified cross-validation
- Generates confusion matrices, ROC curves, and PR curves
- Compares models and saves the best one to `models/`

### Example 2: Programmatic Model Training

Use the training class in your own scripts:

```python
from scripts.train_fraud_models import FraudDetectionModelTrainer

# Initialize trainer for credit card dataset
trainer = FraudDetectionModelTrainer(
    dataset_path='data/raw/creditcard.csv',
    target_col='Class',
    dataset_name='Credit Card',
    random_state=42
)

# Step 1: Prepare data with stratified split
trainer.stratified_train_test_split(test_size=0.2)

# Step 2: Train baseline model
baseline_model, baseline_metrics = trainer.train_baseline_model()

# Step 3: Train ensemble model
ensemble_model, ensemble_metrics = trainer.train_ensemble_model(model_type='xgboost')

# Step 4: Cross-validation
cv_results = trainer.perform_cross_validation(k_folds=5)

# Step 5: Compare and select best model
best_name, best_model, comparison_df = trainer.compare_models()

# Save the best model
trainer.save_model(best_model, best_name)
```

### Example 3: Data Preprocessing Workflow

Use individual preprocessing components:

```python
from src.data_loading import load_data
from src.preprocessing import train_test_split_data, scale_features, handle_class_imbalance

# Load data
df = load_data('data/raw/Fraud_Data.csv')

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split_data(
    df, target_col='class', test_size=0.2, stratify=True
)

# Scale features
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Handle class imbalance with SMOTE (on training data only!)
X_balanced, y_balanced = handle_class_imbalance(
    X_train_scaled, y_train, method='smote', random_state=42
)
```

### Example 4: Load and Use a Trained Model

```python
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load('models/credit_card_xgboost.pkl')

# Load scaler (if saved separately)
# scaler = joblib.load('models/scaler.pkl')

# Prepare new transaction data
new_transaction = pd.DataFrame({
    'V1': [1.2],
    'V2': [-0.5],
    # ... other features
})

# Scale features (use the same scaler from training)
# new_transaction_scaled = scaler.transform(new_transaction)

# Predict
prediction = model.predict(new_transaction)
probability = model.predict_proba(new_transaction)

print(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Legitimate'}")
print(f"Fraud Probability: {probability[0][1]:.4f}")
```

### Example 5: Quick Data Exploration

```python
from src.data_loading import load_data
from src.eda_utils import analyze_class_distribution

# Load dataset
df = load_data('data/raw/creditcard.csv')

# Analyze class distribution
class_dist = analyze_class_distribution(
    df, target_col='Class', dataset_name='Credit Card', visualize=True
)

# Check basic statistics
print(df.describe())
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nClass distribution:\n{df['Class'].value_counts()}")
```

---

## 🔄 Workflow

### Phase 1: Data Exploration & Cleaning ✅
**Location**: `notebooks/eda-fraud-data.ipynb`

1. **Load datasets** using reusable functions
2. **Inspect data quality**
   - Check missing values
   - Identify duplicates
   - Validate data types
3. **Clean data**
   - Remove duplicates
   - Handle missing values
   - Filter invalid entries (age, amounts)
4. **Class distribution analysis**
   - Calculate imbalance ratios
   - Visualize fraud vs legitimate transactions

**Output**: 
- `data/processed/fraud_data_cleaned.csv`
- `data/processed/creditcard_cleaned.csv`

### Phase 2: Feature Engineering ✅
**Location**: `notebooks/eda-fraud-data.ipynb` (Sections 9-10)

1. **Time-based features** (Fraud_Data)
   - Hour of day, day of week
   - Weekend/night flags
   - Time since signup
2. **Behavioral features**
   - Quick purchase detection
   - Rapid transaction velocity
3. **Statistical features** (CreditCard)
   - Mean, std, min, max of V features
   - Log-transformed amounts
4. **Geolocation integration**
   - IP to country mapping
   - Fraud rate by country

**Output**: Engineered features added to cleaned datasets

### Phase 3: Data Preprocessing ✅
**Location**: `notebooks/eda-fraud-data.ipynb` (Sections 10-11)

1. **Train-test split** (80-20 with stratification)
2. **Feature scaling** (StandardScaler on numeric features)
3. **Categorical encoding** (One-hot encoding)
4. **Class imbalance handling**
   - SMOTE (Synthetic Minority Over-sampling)
   - Random Under-sampling (alternative)

**Output**:
- `data/processed/fraud_X_train_smote.csv`
- `data/processed/fraud_y_train_smote.csv`
- `data/processed/fraud_X_test.csv`
- `data/processed/fraud_y_test.csv`

### Phase 4: Model Training ✅
**Location**: `scripts/train_fraud_models.py`

1. **Baseline Model**: Logistic Regression with `class_weight='balanced'`
2. **Ensemble Model**: XGBoost with hyperparameter tuning
3. **Stratified K-Fold Cross-Validation** (k=5)
4. **Model Comparison**: Side-by-side performance evaluation
5. **Model Selection**: Best model based on AUC-PR and interpretability

**Run:**
```bash
python scripts/train_fraud_models.py
```

**Output**:
- Trained models saved in `models/`
- Evaluation metrics and visualizations in `temp_charts/`
- Comprehensive performance reports in console

### Phase 5: Model Evaluation ✅
**Integrated in**: `scripts/train_fraud_models.py`

- Confusion Matrix
- Precision, Recall, F1-Score
- ROC-AUC Curve
- PR-AUC (Area Under Precision-Recall Curve) - primary metric for imbalanced data
- Cross-validation results with mean ± std
- Model comparison and recommendations

---

## 📈 Results Organization

### Processed Data Files
All preprocessed data is saved in `data/processed/` with clear naming conventions:

| File | Description |
|------|-------------|
| `fraud_data_cleaned.csv` | Cleaned e-commerce transaction data |
| `creditcard_cleaned.csv` | Cleaned credit card transaction data |
| `fraud_X_train_smote.csv` | SMOTE-balanced training features |
| `fraud_y_train_smote.csv` | SMOTE-balanced training labels |
| `fraud_X_test.csv` | Test features (imbalanced - real distribution) |
| `fraud_y_test.csv` | Test labels |

### Key Findings (from EDA)

#### Fraud_Data (E-commerce)
- **Imbalance Ratio**: ~10:1 (legitimate vs fraud)
- **High-risk patterns**:
  - Very quick purchases after signup (<5 min)
  - Late night transactions
  - Multiple rapid transactions from same user
  - Certain countries show higher fraud rates

#### CreditCard (Bank)
- **Imbalance Ratio**: ~580:1 (extreme imbalance!)
- **High-risk patterns**:
  - Unusual transaction amounts
  - Atypical V-feature values (PCA components)
  - Transaction timing patterns

---

## ✨ Key Features

### Production-Grade Code
- ✅ **Reusable functions** with docstrings and error handling
- ✅ **Modular design** for easy maintenance and testing
- ✅ **Comprehensive logging** at every step
- ✅ **Data validation** to catch errors early

### Data Leakage Prevention
- ✅ **Proper train-test split** before any transformation
- ✅ **Scaling fitted on training data only**
- ✅ **SMOTE applied to training set only**
- ✅ **Test set maintains real-world distribution**

### Class Imbalance Solutions
- ✅ **SMOTE** for synthetic oversampling
- ✅ **Random Under-sampling** as alternative
- ✅ **Visualizations** showing before/after balancing
- ✅ **Multiple strategies** for comparison

### Comprehensive EDA
- ✅ **Parallel analysis** of both datasets
- ✅ **Univariate and bivariate** visualizations
- ✅ **Geolocation analysis** (fraud by country)
- ✅ **Clear documentation** of findings

---

## 🛠 Technologies Used

### Core Libraries
- **pandas** 1.5+ - Data manipulation and analysis
- **numpy** 1.23+ - Numerical computing
- **scikit-learn** 1.2+ - Machine learning algorithms and preprocessing
- **imbalanced-learn** 0.10+ - Handling class imbalance
- **xgboost** 2.0+ - Gradient boosting for ensemble models
- **joblib** 1.3+ - Model serialization and persistence

### Visualization
- **matplotlib** 3.7+ - Static plots and charts
- **seaborn** 0.12+ - Statistical data visualization

### Development Tools
- **Jupyter Notebook** - Interactive exploration
- **Git** - Version control
- **pytest** - Unit testing framework

---

## 📝 Best Practices Followed

### Code Best Practices
1. **Modular Design**
   - ✅ Reusable functions in `src/` modules (not inline in notebooks)
   - ✅ Functions with comprehensive docstrings and error handling
   - ✅ Notebooks call reusable functions for consistency
   - ✅ Easy to test and maintain

2. **Error Handling**
   - ✅ Try/except blocks with informative error messages
   - ✅ Validation of data before processing
   - ✅ Graceful handling of missing files or invalid data

3. **Data Handling**
   - ✅ Raw data never modified (read-only)
   - ✅ All transformations documented
   - ✅ Reproducible preprocessing pipeline
   - ✅ Proper train-test split to prevent data leakage

4. **Code Quality**
   - ✅ Functions over repeated code
   - ✅ Clear variable naming
   - ✅ Comprehensive comments and docstrings
   - ✅ Consistent code style

5. **Version Control**
   - ✅ `.gitignore` excludes large files and sensitive data
   - ✅ Meaningful commit messages
   - ✅ Separate branches for features

6. **Documentation**
   - ✅ README with complete setup instructions
   - ✅ Inline code documentation
   - ✅ Notebook markdown explanations
   - ✅ Function docstrings with parameters and returns

---

## 🤝 Contributing

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Workflow
1. Create issue describing the problem/feature
2. Assign yourself and link to project board
3. Write tests for new functionality
4. Ensure all tests pass
5. Update documentation
6. Submit PR for review

---

## 📞 Contact & Support

**Project Maintainer**: Adey Innovations Inc.  
**Repository**: [GitHub Link](https://github.com/dnNatnael/improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions)

For questions or issues:
1. Check existing [Issues](../../issues)
2. Create a new issue with detailed description
3. Tag with appropriate labels (bug, enhancement, question)

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **10 Academy** - For the training program and project guidance
- **Adey Innovations Inc.** - For the business case and datasets
- **Kaggle** - For the credit card fraud dataset
- **Open Source Community** - For the amazing libraries used in this project

---

## 🚀 Quick Start Summary

```bash
# 1. Clone and navigate
git clone <repo-url>
cd improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions

# 2. Set up environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# 3. Run EDA notebook (optional - for exploration)
jupyter notebook notebooks/eda-fraud-data.ipynb

# 4. Train fraud detection models
python scripts/train_fraud_models.py

# 5. Check results
# - Models: models/
# - Visualizations: temp_charts/
# - Processed data: data/processed/
```

**Typical Workflow**: 
1. **Explore Data**: Run `notebooks/eda-fraud-data.ipynb` to understand the datasets
2. **Train Models**: Run `python scripts/train_fraud_models.py` to build and evaluate models
3. **Review Results**: Check `temp_charts/` for visualizations and console output for metrics
4. **Deploy**: Use saved models from `models/` for production predictions

---

**Happy Fraud Hunting! 🕵️‍♂️💳**
