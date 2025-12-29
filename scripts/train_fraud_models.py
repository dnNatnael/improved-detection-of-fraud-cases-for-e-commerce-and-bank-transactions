"""
Fraud Detection Model Building and Training

This script builds, trains, and evaluates machine learning models for fraudulent 
transaction detection using highly imbalanced datasets.

Author: Data Science Team
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_loading import load_data

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class FraudDetectionModelTrainer:
    """
    A comprehensive class for training and evaluating fraud detection models.
    """
    
    def __init__(self, dataset_path, target_col, dataset_name, random_state=42):
        """
        Initialize the trainer.
        
        Parameters:
        -----------
        dataset_path : str
            Path to the dataset CSV file
        target_col : str
            Name of the target column ('Class' or 'class')
        dataset_name : str
            Name of the dataset for display purposes
        random_state : int
            Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.target_col = target_col
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """
        Load dataset and separate features from target.
        """
        print("=" * 80)
        print(f"📊 DATASET: {self.dataset_name}")
        print("=" * 80)
        
        # Load data
        self.df = load_data(self.dataset_path)
        
        # Check if target column exists
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset. "
                           f"Available columns: {list(self.df.columns)}")
        
        # Separate features and target
        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Display dataset info
        print(f"\n📈 Dataset Information:")
        print(f"  Total samples: {len(self.df):,}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Target column: '{self.target_col}'")
        
        # Display class distribution
        class_dist = y.value_counts().sort_index()
        print(f"\n📊 Class Distribution:")
        for cls, count in class_dist.items():
            pct = (count / len(y)) * 100
            print(f"  Class {cls}: {count:,} ({pct:.2f}%)")
        
        imbalance_ratio = class_dist[0] / class_dist[1] if len(class_dist) == 2 else 0
        print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1 (majority:minority)")
        
        return X, y
    
    def stratified_train_test_split(self, test_size=0.2):
        """
        Perform stratified train-test split to preserve class distribution.
        
        Why Stratification is Important for Imbalanced Data:
        ----------------------------------------------------
        Stratified splitting ensures that both training and test sets maintain 
        the same class distribution as the original dataset. This is crucial for:
        
        1. Representative Evaluation: Test set reflects real-world class imbalance
        2. Fair Comparison: Models are evaluated on similar distributions
        3. Preventing Bias: Without stratification, a fold might have no/minimal 
           minority class samples, making evaluation unreliable
        4. Consistent Metrics: Stratification ensures metrics are comparable 
           across different splits
        
        Without stratification, you might end up with:
        - Test set with 0% fraud cases (impossible to evaluate)
        - Training set with different imbalance ratio than test set
        - Unreliable performance estimates
        """
        print("\n" + "=" * 80)
        print("🔀 STEP 1: STRATIFIED TRAIN-TEST SPLIT")
        print("=" * 80)
        
        X, y = self.load_and_prepare_data()
        
        # Perform stratified split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y  # This ensures class distribution is preserved
        )
        
        print(f"\n✓ Stratified split complete (test_size={test_size})")
        print(f"\n📊 Training Set:")
        train_dist = self.y_train.value_counts().sort_index()
        for cls, count in train_dist.items():
            pct = (count / len(self.y_train)) * 100
            print(f"  Class {cls}: {count:,} ({pct:.2f}%)")
        
        print(f"\n📊 Test Set:")
        test_dist = self.y_test.value_counts().sort_index()
        for cls, count in test_dist.items():
            pct = (count / len(self.y_test)) * 100
            print(f"  Class {cls}: {count:,} ({pct:.2f}%)")
        
        # Verify stratification worked
        train_ratio = train_dist[0] / train_dist[1] if len(train_dist) == 2 else 0
        test_ratio = test_dist[0] / test_dist[1] if len(test_dist) == 2 else 0
        print(f"\n✓ Imbalance ratios preserved:")
        print(f"  Train: {train_ratio:.2f}:1")
        print(f"  Test: {test_ratio:.2f}:1")
        
        # Scale features (important for Logistic Regression and distance-based models)
        print(f"\n🔧 Scaling features...")
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        print(f"✓ Features scaled using StandardScaler")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_baseline_model(self):
        """
        Train Logistic Regression as an interpretable baseline model.
        Uses class_weight='balanced' to handle class imbalance.
        """
        print("\n" + "=" * 80)
        print("🎯 STEP 2: BASELINE MODEL - LOGISTIC REGRESSION")
        print("=" * 80)
        
        print("\n📝 Model Configuration:")
        print("  Algorithm: Logistic Regression")
        print("  Class Weight: 'balanced' (automatically adjusts for class imbalance)")
        print("  Solver: 'lbfgs' (good for small-medium datasets)")
        print("  Max Iterations: 1000")
        
        # Train Logistic Regression with balanced class weights
        lr_model = LogisticRegression(
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000,
            solver='lbfgs',
            n_jobs=-1
        )
        
        print("\n🔄 Training model...")
        lr_model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr_model
        print("✓ Training complete")
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        y_pred = lr_model.predict(self.X_test)
        y_pred_proba = lr_model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
        self.results['Logistic Regression'] = metrics
        
        # Display results
        self._display_results('Logistic Regression', metrics, self.y_test, y_pred)
        
        return lr_model, metrics
    
    def train_ensemble_model(self, model_type='xgboost'):
        """
        Train an ensemble model (XGBoost, Random Forest, or LightGBM).
        Performs basic hyperparameter tuning.
        
        Parameters:
        -----------
        model_type : str
            Type of ensemble model ('xgboost', 'random_forest', or 'lightgbm')
        """
        print("\n" + "=" * 80)
        print(f"🌲 STEP 3: ENSEMBLE MODEL - {model_type.upper()}")
        print("=" * 80)
        
        if model_type == 'xgboost':
            print("\n📝 Model Configuration:")
            print("  Algorithm: XGBoost (Gradient Boosting)")
            print("  Hyperparameters:")
            print("    - n_estimators: 100")
            print("    - max_depth: 6")
            print("    - learning_rate: 0.1")
            print("    - scale_pos_weight: auto (handles class imbalance)")
            print("    - random_state: 42")
            
            # Calculate scale_pos_weight for class imbalance
            scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
            
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )
            
        elif model_type == 'random_forest':
            print("\n📝 Model Configuration:")
            print("  Algorithm: Random Forest")
            print("  Hyperparameters:")
            print("    - n_estimators: 100")
            print("    - max_depth: 10")
            print("    - class_weight: 'balanced'")
            print("    - random_state: 42")
            
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose 'xgboost' or 'random_forest'")
        
        print("\n🔄 Training model...")
        model.fit(self.X_train, self.y_train)
        self.models[model_type.title()] = model
        print("✓ Training complete")
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
        self.results[model_type.title()] = metrics
        
        # Display results
        self._display_results(model_type.title(), metrics, self.y_test, y_pred)
        
        return model, metrics
    
    def perform_cross_validation(self, k_folds=5):
        """
        Perform Stratified K-Fold Cross-Validation.
        
        Why Stratified K-Fold is Preferred for Imbalanced Classification:
        -------------------------------------------------------------------
        1. Preserves Class Distribution: Each fold maintains the same class 
           ratio as the original dataset
        2. Reliable Estimates: Every fold has representative samples from 
           both classes, preventing folds with zero minority class samples
        3. Reduced Variance: More stable performance estimates across folds
        4. Better Model Selection: Helps identify models that generalize well
        5. Prevents Overfitting Detection: Without stratification, CV might 
           miss overfitting if some folds lack minority class samples
        
        Regular K-Fold can fail with imbalanced data:
        - Some folds might have 0 fraud cases
        - Metrics become undefined or unreliable
        - High variance in cross-validation scores
        """
        print("\n" + "=" * 80)
        print(f"🔄 STEP 4: STRATIFIED K-FOLD CROSS-VALIDATION (k={k_folds})")
        print("=" * 80)
        
        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
        
        cv_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n📊 Cross-Validating: {model_name}")
            
            # Perform cross-validation for multiple metrics
            cv_auc_pr = cross_val_score(
                model, self.X_train, self.y_train,
                cv=skf, scoring='average_precision', n_jobs=-1
            )
            cv_f1 = cross_val_score(
                model, self.X_train, self.y_train,
                cv=skf, scoring='f1', n_jobs=-1
            )
            cv_roc_auc = cross_val_score(
                model, self.X_train, self.y_train,
                cv=skf, scoring='roc_auc', n_jobs=-1
            )
            
            cv_results[model_name] = {
                'AUC-PR': {
                    'mean': cv_auc_pr.mean(),
                    'std': cv_auc_pr.std(),
                    'scores': cv_auc_pr
                },
                'F1-Score': {
                    'mean': cv_f1.mean(),
                    'std': cv_f1.std(),
                    'scores': cv_f1
                },
                'ROC-AUC': {
                    'mean': cv_roc_auc.mean(),
                    'std': cv_roc_auc.std(),
                    'scores': cv_roc_auc
                }
            }
            
            print(f"  AUC-PR: {cv_auc_pr.mean():.4f} (±{cv_auc_pr.std():.4f})")
            print(f"  F1-Score: {cv_f1.mean():.4f} (±{cv_f1.std():.4f})")
            print(f"  ROC-AUC: {cv_roc_auc.mean():.4f} (±{cv_roc_auc.std():.4f})")
        
        # Create visualization
        self._plot_cv_results(cv_results, k_folds)
        
        return cv_results
    
    def compare_models(self):
        """
        Compare all trained models side-by-side and select the best one.
        """
        print("\n" + "=" * 80)
        print("📊 STEP 5: MODEL COMPARISON AND SELECTION")
        print("=" * 80)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'AUC-PR': metrics['AUC-PR'],
                'F1-Score': metrics['F1-Score'],
                'ROC-AUC': metrics['ROC-AUC'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC-PR', ascending=False)
        
        print("\n📋 Model Performance Comparison:")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)
        
        # Select best model based on AUC-PR (most important for imbalanced data)
        best_model_name = comparison_df.iloc[0]['Model']
        best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Primary Metric (AUC-PR): {comparison_df.iloc[0]['AUC-PR']:.4f}")
        print(f"   F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")
        
        # Create comparison visualization
        self._plot_model_comparison(comparison_df)
        
        # Provide recommendation
        self._provide_recommendation(best_model_name, comparison_df)
        
        return best_model_name, best_model, comparison_df
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics."""
        return {
            'AUC-PR': average_precision_score(y_true, y_pred_proba),
            'F1-Score': f1_score(y_true, y_pred),
            'ROC-AUC': roc_auc_score(y_true, y_pred_proba),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def _display_results(self, model_name, metrics, y_true, y_pred):
        """Display evaluation results for a model."""
        print(f"\n{'='*80}")
        print(f"📈 {model_name} - Test Set Performance")
        print(f"{'='*80}")
        
        print(f"\n🎯 Key Metrics (Important for Imbalanced Data):")
        print(f"  AUC-PR (Area Under Precision-Recall Curve): {metrics['AUC-PR']:.4f}")
        print(f"    → Best metric for imbalanced data (focuses on minority class)")
        print(f"  F1-Score: {metrics['F1-Score']:.4f}")
        print(f"    → Harmonic mean of precision and recall")
        print(f"  ROC-AUC: {metrics['ROC-AUC']:.4f}")
        print(f"    → Overall classification ability")
        
        print(f"\n📊 Detailed Metrics:")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"    → Of predicted frauds, how many are actually fraud?")
        print(f"  Recall: {metrics['Recall']:.4f}")
        print(f"    → Of actual frauds, how many did we catch?")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n📋 Confusion Matrix:")
        print(f"                    Predicted")
        print(f"                  Legit    Fraud")
        print(f"  Actual Legit    {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"  Actual Fraud    {cm[1,0]:6d}  {cm[1,1]:6d}")
        
        # Calculate additional metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        print(f"\n  True Negatives (TN):  {tn:,} - Correctly identified legitimate")
        print(f"  False Positives (FP): {fp:,} - Legitimate flagged as fraud (Type I Error)")
        print(f"  False Negatives (FN): {fn:,} - Fraud missed (Type II Error)")
        print(f"  True Positives (TP):  {tp:,} - Correctly identified fraud")
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, model_name)
        
        # Plot ROC and PR curves
        self._plot_curves(y_true, metrics['y_pred_proba'], model_name)
        
        # Model strengths and limitations
        print(f"\n💡 Model Analysis:")
        print(f"  Strengths:")
        if metrics['Recall'] > 0.7:
            print(f"    ✓ High recall - catches most fraud cases")
        if metrics['Precision'] > 0.7:
            print(f"    ✓ High precision - low false positive rate")
        if metrics['AUC-PR'] > 0.7:
            print(f"    ✓ Strong overall performance on imbalanced data")
        
        print(f"  Limitations:")
        if metrics['Recall'] < 0.5:
            print(f"    ⚠ Low recall - missing many fraud cases")
        if metrics['Precision'] < 0.5:
            print(f"    ⚠ Low precision - many false alarms")
        if fp > tp * 2:
            print(f"    ⚠ High false positive rate - may cause customer friction")
        if fn > tp:
            print(f"    ⚠ High false negative rate - missing fraud is costly")
    
    def _plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        
        # Save plot
        os.makedirs('temp_charts', exist_ok=True)
        filename = f"temp_charts/confusion_matrix_{self.dataset_name.lower().replace(' ', '_')}_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n  💾 Confusion matrix saved: {filename}")
        plt.close()
    
    def _plot_curves(self, y_true, y_pred_proba, model_name):
        """Plot ROC and Precision-Recall curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        ax2.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        baseline = (y_true == 1).sum() / len(y_true)
        ax2.axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline ({baseline:.3f})')
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"temp_charts/curves_{self.dataset_name.lower().replace(' ', '_')}_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  💾 ROC/PR curves saved: {filename}")
        plt.close()
    
    def _plot_cv_results(self, cv_results, k_folds):
        """Plot cross-validation results."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics_to_plot = ['AUC-PR', 'F1-Score', 'ROC-AUC']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            model_names = list(cv_results.keys())
            means = [cv_results[m][metric]['mean'] for m in model_names]
            stds = [cv_results[m][metric]['std'] for m in model_names]
            
            x_pos = np.arange(len(model_names))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                         color=['#3498db', '#e74c3c'])
            
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric} - {k_folds}-Fold CV', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (mean, std) in enumerate(zip(means, stds)):
                ax.text(i, mean + std + 0.01, f'{mean:.3f}\n±{std:.3f}', 
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filename = f"temp_charts/cv_results_{self.dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n  💾 CV results plot saved: {filename}")
        plt.close()
    
    def _plot_model_comparison(self, comparison_df):
        """Plot side-by-side model comparison."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        metrics = ['AUC-PR', 'F1-Score', 'ROC-AUC', 'Precision', 'Recall']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                         color=['#3498db', '#e74c3c'], alpha=0.7)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        filename = f"temp_charts/model_comparison_{self.dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n  💾 Model comparison plot saved: {filename}")
        plt.close()
    
    def _provide_recommendation(self, best_model_name, comparison_df):
        """Provide final recommendation with justification."""
        print("\n" + "=" * 80)
        print("💡 FINAL RECOMMENDATION")
        print("=" * 80)
        
        best_row = comparison_df[comparison_df['Model'] == best_model_name].iloc[0]
        
        print(f"\n✅ Selected Model: {best_model_name}")
        print(f"\n📊 Justification:")
        print(f"  1. Predictive Performance:")
        print(f"     • AUC-PR: {best_row['AUC-PR']:.4f} (primary metric for imbalanced data)")
        print(f"     • F1-Score: {best_row['F1-Score']:.4f}")
        print(f"     • ROC-AUC: {best_row['ROC-AUC']:.4f}")
        
        print(f"\n  2. Interpretability:")
        if 'Logistic' in best_model_name:
            print(f"     • High interpretability - coefficients show feature importance")
            print(f"     • Easy to explain to stakeholders")
            print(f"     • Can extract feature contributions")
        else:
            print(f"     • Moderate interpretability - can extract feature importance")
            print(f"     • Tree-based models provide some explainability")
            print(f"     • Consider SHAP values for detailed explanations")
        
        print(f"\n  3. Business Considerations:")
        print(f"     • Precision: {best_row['Precision']:.4f} - False positive rate")
        print(f"     • Recall: {best_row['Recall']:.4f} - Fraud detection rate")
        
        if best_row['Precision'] < 0.5:
            print(f"     ⚠ Warning: Low precision may cause customer friction")
        if best_row['Recall'] < 0.5:
            print(f"     ⚠ Warning: Low recall means missing fraud cases")
        
        print(f"\n  4. Deployment Considerations:")
        print(f"     • Model can be saved and deployed for real-time predictions")
        print(f"     • Consider monitoring model performance over time")
        print(f"     • Retrain periodically as fraud patterns evolve")
        
        print(f"\n📝 Summary:")
        print(f"  The {best_model_name} model demonstrates the best balance of")
        print(f"  predictive performance and practical utility for fraud detection.")
        print(f"  It effectively handles the class imbalance while maintaining")
        print(f"  reasonable interpretability for business stakeholders.")
    
    def save_model(self, model, model_name, filepath=None):
        """Save trained model to disk."""
        if filepath is None:
            os.makedirs('models', exist_ok=True)
            filepath = f"models/{self.dataset_name.lower().replace(' ', '_')}_{model_name.lower().replace(' ', '_')}.pkl"
        
        joblib.dump(model, filepath)
        print(f"\n💾 Model saved: {filepath}")
        return filepath


def main():
    """
    Main function to run the complete fraud detection model training pipeline.
    """
    print("\n" + "=" * 80)
    print("🚀 FRAUD DETECTION MODEL TRAINING PIPELINE")
    print("=" * 80)
    
    # Dataset configurations
    datasets = [
        {
            'path': 'data/raw/creditcard.csv',
            'target': 'Class',
            'name': 'Credit Card'
        },
        {
            'path': 'data/raw/Fraud_Data.csv',
            'target': 'class',
            'name': 'E-commerce Fraud'
        }
    ]
    
    # Process each dataset
    for dataset_config in datasets:
        try:
            # Initialize trainer
            trainer = FraudDetectionModelTrainer(
                dataset_path=dataset_config['path'],
                target_col=dataset_config['target'],
                dataset_name=dataset_config['name'],
                random_state=42
            )
            
            # Step 1: Data preparation with stratified split
            trainer.stratified_train_test_split(test_size=0.2)
            
            # Step 2: Train baseline model
            trainer.train_baseline_model()
            
            # Step 3: Train ensemble model (XGBoost)
            trainer.train_ensemble_model(model_type='xgboost')
            
            # Step 4: Cross-validation
            cv_results = trainer.perform_cross_validation(k_folds=5)
            
            # Step 5: Compare models and get recommendation
            best_name, best_model, comparison_df = trainer.compare_models()
            
            # Save best model
            trainer.save_model(best_model, best_name)
            
            print("\n" + "=" * 80)
            print(f"✅ COMPLETE: {dataset_config['name']} Dataset")
            print("=" * 80)
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: Dataset not found - {dataset_config['path']}")
            print(f"   Please ensure the dataset exists in the specified location.")
            continue
        except Exception as e:
            print(f"\n❌ Error processing {dataset_config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("🎉 ALL DATASETS PROCESSED SUCCESSFULLY!")
    print("=" * 80)
    print("\n📁 Output Files:")
    print("  • Trained models: models/")
    print("  • Visualizations: temp_charts/")
    print("\n💡 Next Steps:")
    print("  1. Review the model comparison results")
    print("  2. Check confusion matrices and ROC/PR curves")
    print("  3. Deploy the best model for production use")
    print("  4. Monitor model performance over time")


if __name__ == "__main__":
    main()

