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
│   └── preprocess_data.py    # Main preprocessing pipeline script
│
├── src/                       # Reusable Python modules
│   ├── data_loading.py       # Data loading with error handling
│   ├── data_cleaning.py      # Data cleaning and standardization
│   ├── feature_engineering.py # Feature engineering and geolocation
│   ├── eda_utils.py          # EDA utilities and visualizations
│   └── preprocessing.py      # Preprocessing pipeline (scaling, encoding, SMOTE)
│
├── tests/                     # Unit tests
│   ├── (future test files)
│   └── __init__.py
│
├── models/                    # Saved trained models
│   └── (future: .pkl or .joblib files)
│
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
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

### Option 2: Run Python Scripts (Future: For Production)

```bash
# Example: Future training script
python scripts/train_model.py --dataset fraud_data --model xgboost

# Example: Future evaluation script
python scripts/evaluate_model.py --model fraud_model.pkl
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

### Phase 4: Model Training (Future)
**Location**: `scripts/train_model.py` (To be implemented)

- Logistic Regression (baseline)
- Random Forest
- XGBoost
- Neural Networks

### Phase 5: Model Evaluation (Future)
**Location**: `scripts/evaluate_model.py` (To be implemented)

- Confusion Matrix
- Precision, Recall, F1-Score
- ROC-AUC Curve
- PR-AUC (important for imbalanced data)

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

### Visualization
- **matplotlib** 3.7+ - Static plots and charts
- **seaborn** 0.12+ - Statistical data visualization

### Development Tools
- **Jupyter Notebook** - Interactive exploration
- **Git** - Version control
- **pytest** (future) - Unit testing

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
pip install -r requirements.txt

# 3. Run EDA notebook
jupyter notebook notebooks/eda-fraud-data.ipynb

# 4. Check processed data
ls data/processed/
```

**Next Steps**: 
1. Run the EDA notebook to familiarize yourself with the data
2. Review the processed datasets in `data/processed/`
3. Start building your fraud detection models!

---

**Happy Fraud Hunting! 🕵️‍♂️💳**
