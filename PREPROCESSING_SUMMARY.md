# Data Preprocessing & EDA Summary

## ✅ Completed Tasks

### 1. Data Cleaning ✓
- **Handled Missing Values**: Comprehensive analysis and strategic imputation/dropping based on percentage
- **Removed Duplicates**: Identified and removed duplicate records
- **Data Type Standardization**: Converted columns to appropriate types (datetime, numeric, etc.)
- **Invalid Data Filtering**: Filtered invalid ages (0-120) and negative transaction values

**Implementation**: `src/data_cleaning.py`
- `clean_data()`: Comprehensive cleaning with configurable options
- `standardize_datatypes()`: Type standardization for both datasets

### 2. Exploratory Data Analysis (EDA) ✓

#### Univariate Analysis
- Distribution of numerical features (purchase_value, age, Amount, Time)
- Frequency of categorical features (source, browser, sex)
- Visualizations using histograms, bar charts

#### Bivariate Analysis
- Relationship between features and fraud label (boxplots, violin plots)
- Statistical comparisons by class
- Correlation analysis with visualizations

#### Class Distribution
- Quantified fraud vs non-fraud transactions
- Visualized class imbalance (bar charts, pie charts)
- Explained why imbalance is problematic for ML models
- Calculated imbalance ratios

**Implementation**: `src/eda_utils.py`
- `analyze_class_distribution()`: Comprehensive class analysis with visualizations
- `analyze_missing_values()`: Missing value analysis
- `analyze_duplicates()`: Duplicate detection
- `get_data_summary()`: Overall data summary

### 3. Geolocation Integration ✓
- **IP Address Conversion**: Converted IP addresses to integer format
- **Range-Based Merge**: Performed efficient range-based merge between Fraud_Data.csv and IpAddress_to_Country.csv
- **Country Feature**: Added country feature to fraud dataset
- **Fraud Pattern Analysis**: Analyzed fraud rates by country with visualizations
  - Top countries by fraud rate
  - Top countries by transaction volume with fraud rates

**Implementation**: `src/feature_engineering.py`
- `convert_ip_to_int()`: IP to integer conversion
- `merge_geolocation()`: Range-based geolocation merge

### 4. Feature Engineering ✓

#### Fraud_Data Features:
- **Time-Based Features**:
  - `hour_of_day`: Purchase hour (0-23)
  - `day_of_week`: Purchase day (0=Monday, 6=Sunday)
  - `is_weekend`: Weekend flag (1 if Saturday/Sunday)
  - `is_night`: Night flag (1 if 10 PM - 6 AM)
  - `time_since_signup`: Hours between signup and purchase
  
- **Behavioral Features**:
  - `quick_purchase`: 1 if purchase within 5 minutes of signup
  - `transaction_count`: Number of transactions per user
  - `rapid_transactions`: 1 if transaction within 1 hour of previous

#### CreditCard Features:
- **Statistical Aggregations**:
  - `features_mean`: Mean of V-features
  - `features_std`: Standard deviation of V-features
  - `features_min`: Minimum of V-features
  - `features_max`: Maximum of V-features
  
- **Transformations**:
  - `log_amount`: Log-transformed transaction amount
  - `hour_sin/cos`: Cyclical encoding of transaction hour

**Implementation**: `src/feature_engineering.py`
- `engineer_features()`: Main feature engineering function
- `_engineer_fraud_features()`: Fraud-specific features
- `_engineer_creditcard_features()`: Credit card-specific features

**Justification for Each Feature**:
- Time-based features: Fraudsters often operate during off-hours (nights, weekends)
- Quick purchase: Fraudulent accounts often make purchases immediately after signup
- Rapid transactions: Multiple transactions in short time indicate suspicious behavior
- Statistical aggregations: Capture overall transaction behavior patterns
- Log transformation: Reduces skewness in transaction amounts

### 5. Data Transformation ✓
- **Feature Scaling**: StandardScaler applied to numeric features
  - Fit on training data only (prevents data leakage)
  - Transform both training and test sets
  
- **Categorical Encoding**: One-Hot Encoding ready for categorical variables
  - Can be applied with `encode_categorical_features()` function

**Implementation**: `src/preprocessing.py`
- `scale_features()`: StandardScaler or MinMaxScaler
- `encode_categorical_features()`: One-hot encoding

### 6. Class Imbalance Handling ✓
- **SMOTE (Synthetic Minority Over-sampling)**: Applied to training data only
- **Justification**: 
  - Creates synthetic samples in feature space (better than simple oversampling)
  - Preserves original data distribution
  - Only applied to training data to prevent data leakage
  - Test set maintains real-world imbalanced distribution
  
- **Before/After Comparison**: Visualizations showing class distributions
- **Alternative Method**: Random Under-sampling also available (but not recommended due to data loss)

**Implementation**: `src/preprocessing.py`
- `handle_class_imbalance()`: SMOTE or undersampling
- `train_test_split_data()`: Stratified train-test split (80-20)

## 📁 File Structure

```
src/
├── data_loading.py          # Data loading with error handling
├── data_cleaning.py         # Data cleaning and standardization
├── feature_engineering.py   # Feature engineering and geolocation
├── eda_utils.py             # EDA utilities and visualizations
└── preprocessing.py         # Preprocessing pipeline (scaling, encoding, SMOTE)

notebooks/
└── eda-fraud-data.ipynb     # Comprehensive EDA and preprocessing notebook

data/
├── raw/                     # Original datasets (DO NOT MODIFY)
│   ├── Fraud_Data.csv
│   ├── creditcard.csv
│   └── IpAddress_to_Country.csv
└── processed/               # Cleaned and preprocessed data
    ├── fraud_data_cleaned.csv
    ├── creditcard_cleaned.csv
    ├── fraud_X_train_smote.csv
    ├── fraud_y_train_smote.csv
    ├── fraud_X_test.csv
    ├── fraud_y_test.csv
    ├── creditcard_X_train_smote.csv
    ├── creditcard_y_train_smote.csv
    ├── creditcard_X_test.csv
    └── creditcard_y_test.csv
```

## 🚀 How to Use

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
jupyter notebook notebooks/eda-fraud-data.ipynb
```

Then run all cells sequentially (Kernel → Restart & Run All)

### 3. Output Files
After running the notebook, you'll have:
- Cleaned datasets in `data/processed/`
- Train/test splits with SMOTE-balanced training data
- Test sets with real-world imbalanced distribution

## 📊 Key Insights

### Fraud_Data (E-commerce)
- **Class Imbalance**: ~10:1 (legitimate:fraud)
- **High-Risk Patterns Identified**:
  - Very quick purchases after signup (<5 min)
  - Late night transactions
  - Weekend transactions
  - Multiple rapid transactions from same user
  - Certain countries show higher fraud rates

### CreditCard (Bank)
- **Class Imbalance**: ~580:1 (extreme imbalance!)
- **High-Risk Patterns Identified**:
  - Unusual transaction amounts
  - Atypical V-feature values (PCA components)
  - Transaction timing patterns

## ✅ Best Practices Implemented

1. **Reusable Functions**: All code in `src/` modules for reusability
2. **Error Handling**: Comprehensive error handling in all functions
3. **Data Leakage Prevention**: 
   - Train-test split BEFORE any transformation
   - Scaling fitted on training data only
   - SMOTE applied to training data only
4. **Documentation**: Clear docstrings and inline comments
5. **Reproducibility**: Random seeds set for all random operations
6. **Comprehensive Logging**: Progress messages at each step

## 🎯 Next Steps

1. **Model Training**: Use SMOTE-balanced train sets for model training
2. **Model Evaluation**: Evaluate on imbalanced test sets (real-world scenario)
3. **Feature Selection**: Consider correlation analysis for feature selection
4. **Advanced Feature Engineering**: Explore interaction features, polynomial features
5. **Model Interpretability**: Use SHAP values to understand model decisions

---

**Note**: All preprocessing decisions are documented in the notebook with clear justifications. The code follows production-grade best practices and can easily be moved to production pipelines.

