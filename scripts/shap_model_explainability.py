"""
SHAP Model Explainability Analysis for Fraud Detection

This script performs comprehensive SHAP (SHapley Additive exPlanations) analysis
on trained fraud detection models to understand model predictions and derive
actionable business insights.

Author: Data Science Team
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# SHAP imports
try:
    import shap
    shap.initjs()  # Initialize JavaScript visualization support
except ImportError:
    print("❌ SHAP library not installed. Please install it using: pip install shap")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_loading import load_data

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class SHAPFraudExplainer:
    """
    Comprehensive SHAP explainability analysis for fraud detection models.
    """
    
    def __init__(self, model, X_train, X_test, y_test, feature_names, dataset_name="Fraud Detection", random_state=42):
        """
        Initialize the SHAP explainer.
        
        Parameters:
        -----------
        model : sklearn/xgboost model
            Trained fraud detection model
        X_train : pd.DataFrame or np.array
            Training features (used as background for SHAP)
        X_test : pd.DataFrame or np.array
            Test features for explanation
        y_test : pd.Series or np.array
            Test labels
        feature_names : list
            List of feature names
        dataset_name : str
            Name of the dataset for display purposes
        random_state : int
            Random seed for reproducibility
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.dataset_name = dataset_name
        self.random_state = random_state
        
        # Convert to numpy if DataFrames
        if isinstance(X_train, pd.DataFrame):
            self.X_train_array = X_train.values
            self.X_test_array = X_test.values
        else:
            self.X_train_array = X_train
            self.X_test_array = X_test
            
        if isinstance(y_test, pd.Series):
            self.y_test_array = y_test.values
        else:
            self.y_test_array = y_test
        
        # Initialize explainer
        self.explainer = None
        self.shap_values = None
        self.shap_values_expected = None
        
        # Get predictions
        self.y_pred = model.predict(self.X_test_array)
        self.y_pred_proba = model.predict_proba(self.X_test_array)[:, 1]
        
        print(f"\n{'='*80}")
        print(f"📊 SHAP EXPLAINABILITY ANALYSIS: {dataset_name}")
        print(f"{'='*80}")
        print(f"\n✓ Model loaded successfully")
        print(f"✓ Test set: {len(self.y_test_array)} samples")
        print(f"✓ Features: {len(feature_names)}")
    
    def initialize_shap_explainer(self, sample_size=1000):
        """
        Initialize SHAP explainer based on model type.
        
        Parameters:
        -----------
        sample_size : int
            Number of samples to use for TreeExplainer background (speed optimization)
        """
        print(f"\n{'='*80}")
        print("🔧 STEP 1: INITIALIZING SHAP EXPLAINER")
        print(f"{'='*80}")
        
        model_type = type(self.model).__name__
        
        # Use sample of training data for background (speeds up computation)
        if len(self.X_train_array) > sample_size:
            sample_idx = np.random.choice(len(self.X_train_array), sample_size, replace=False)
            X_background = self.X_train_array[sample_idx]
            print(f"✓ Using {sample_size} samples from training data as background")
        else:
            X_background = self.X_train_array
            print(f"✓ Using all {len(X_background)} training samples as background")
        
        # Select appropriate explainer based on model type
        if 'XGB' in model_type or 'XGBoost' in model_type:
            print(f"✓ Model type: XGBoost - Using TreeExplainer (fast)")
            self.explainer = shap.TreeExplainer(self.model, X_background)
            
        elif 'RandomForest' in model_type or 'GradientBoosting' in model_type:
            print(f"✓ Model type: Tree-based - Using TreeExplainer (fast)")
            self.explainer = shap.TreeExplainer(self.model, X_background)
            
        elif 'LogisticRegression' in model_type:
            print(f"✓ Model type: Linear - Using LinearExplainer")
            self.explainer = shap.LinearExplainer(self.model, X_background)
            
        else:
            print(f"⚠ Model type: {model_type} - Using KernelExplainer (slower)")
            print(f"  Note: This may take longer for large datasets")
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
        
        print(f"✓ SHAP explainer initialized successfully")
    
    def compute_shap_values(self, max_samples=500):
        """
        Compute SHAP values for test set.
        
        Parameters:
        -----------
        max_samples : int
            Maximum number of test samples to explain (for performance)
        """
        print(f"\n{'='*80}")
        print("⚙️ STEP 2: COMPUTING SHAP VALUES")
        print(f"{'='*80}")
        
        # Limit samples for faster computation
        if len(self.X_test_array) > max_samples:
            print(f"⚠ Test set has {len(self.X_test_array)} samples")
            print(f"  Computing SHAP for {max_samples} samples (for performance)")
            print(f"  Note: Increase max_samples for full explanation")
            sample_idx = np.random.choice(len(self.X_test_array), max_samples, replace=False)
            X_explain = self.X_test_array[sample_idx]
            self.y_test_sample = self.y_test_array[sample_idx]
            self.y_pred_sample = self.y_pred[sample_idx]
            self.y_pred_proba_sample = self.y_pred_proba[sample_idx]
            self.X_test_sample = self.X_test_array[sample_idx]
        else:
            X_explain = self.X_test_array
            self.y_test_sample = self.y_test_array
            self.y_pred_sample = self.y_pred
            self.y_pred_proba_sample = self.y_pred_proba
            self.X_test_sample = self.X_test_array
        
        print(f"\n🔄 Computing SHAP values...")
        print(f"  This may take a few minutes depending on model complexity...")
        
        # Compute SHAP values
        self.shap_values = self.explainer.shap_values(X_explain)
        
        # Handle binary classification (SHAP returns array for each class)
        if isinstance(self.shap_values, list):
            # For binary classification, use values for positive class (fraud)
            self.shap_values = self.shap_values[1]
        
        # Get expected value (base value)
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, np.ndarray):
                self.shap_values_expected = self.explainer.expected_value[1]
            else:
                self.shap_values_expected = self.explainer.expected_value
        else:
            self.shap_values_expected = np.mean(self.y_pred_proba_sample)
        
        print(f"✓ SHAP values computed successfully")
        print(f"  Shape: {self.shap_values.shape}")
        print(f"  Expected value (base): {self.shap_values_expected:.4f}")
    
    def analyze_builtin_feature_importance(self):
        """
        Extract and visualize model's built-in feature importance.
        """
        print(f"\n{'='*80}")
        print("📊 STEP 3: BUILT-IN FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*80}")
        
        model_type = type(self.model).__name__
        
        # Extract feature importance based on model type
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            importance_type = "Feature Importance"
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
            importance_type = "Absolute Coefficients"
        else:
            print("⚠ Model does not provide feature importance")
            return None
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n📈 Top 10 Most Important Features ({importance_type}):")
        print("="*80)
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:<40} {row['importance']:>10.6f}")
        
        # Visualize top 10 features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(10)
        plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel(f'{importance_type}', fontsize=12, fontweight='bold')
        plt.title(f'Top 10 Feature Importance - {self.dataset_name}\n({importance_type})', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        os.makedirs('temp_charts', exist_ok=True)
        filename = f"temp_charts/builtin_feature_importance_{self.dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n  💾 Feature importance plot saved: {filename}")
        plt.close()
        
        print(f"\n💡 Interpretation:")
        print(f"  • This shows the model's built-in feature importance")
        print(f"  • Higher values indicate features the model relies on more")
        print(f"  • Limitation: This is an aggregate measure and doesn't show")
        print(f"    how features affect individual predictions")
        
        return importance_df
    
    def global_shap_analysis(self):
        """
        Perform global SHAP analysis (overall feature impact).
        """
        print(f"\n{'='*80}")
        print("🌍 STEP 4: GLOBAL SHAP ANALYSIS")
        print(f"{'='*80}")
        
        # Create SHAP summary plot
        print(f"\n📊 Generating SHAP Summary Plot...")
        plt.figure(figsize=(12, 8))
        
        # Use subset for visualization if too many samples
        plot_samples = min(500, len(self.shap_values))
        if len(self.shap_values) > plot_samples:
            idx = np.random.choice(len(self.shap_values), plot_samples, replace=False)
            shap_values_plot = self.shap_values[idx]
            X_test_plot = self.X_test_sample[idx]
        else:
            shap_values_plot = self.shap_values
            X_test_plot = self.X_test_sample
        
        shap.summary_plot(
            shap_values_plot,
            X_test_plot,
            feature_names=self.feature_names,
            show=False,
            max_display=20
        )
        plt.title(f'SHAP Summary Plot - {self.dataset_name}\n(How Features Impact Fraud Predictions)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save plot
        filename = f"temp_charts/shap_summary_plot_{self.dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  💾 SHAP summary plot saved: {filename}")
        plt.close()
        
        # Calculate mean absolute SHAP values (global importance)
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        shap_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        print(f"\n📈 Top 10 Features by Mean Absolute SHAP Value:")
        print("="*80)
        for idx, row in shap_importance_df.head(10).iterrows():
            print(f"  {row['feature']:<40} {row['mean_abs_shap']:>10.6f}")
        
        # Visualize mean absolute SHAP
        plt.figure(figsize=(12, 8))
        top_shap = shap_importance_df.head(15)
        plt.barh(range(len(top_shap)), top_shap['mean_abs_shap'], color='darkgreen')
        plt.yticks(range(len(top_shap)), top_shap['feature'])
        plt.xlabel('Mean |SHAP Value| (Global Importance)', fontsize=12, fontweight='bold')
        plt.title(f'Top 15 Features by SHAP Importance - {self.dataset_name}', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        filename = f"temp_charts/shap_importance_{self.dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  💾 SHAP importance plot saved: {filename}")
        plt.close()
        
        # Analyze feature impact direction
        print(f"\n📊 Feature Impact Analysis:")
        print("="*80)
        
        # Features that increase fraud probability (positive SHAP)
        mean_shap = self.shap_values.mean(axis=0)
        fraud_increasing = pd.DataFrame({
            'feature': self.feature_names,
            'mean_shap': mean_shap
        }).sort_values('mean_shap', ascending=False).head(10)
        
        print(f"\n🔴 Top Features that INCREASE Fraud Probability:")
        for idx, row in fraud_increasing.iterrows():
            if row['mean_shap'] > 0:
                print(f"  {row['feature']:<40} {row['mean_shap']:>+10.6f}")
        
        # Features that decrease fraud probability (negative SHAP)
        fraud_decreasing = pd.DataFrame({
            'feature': self.feature_names,
            'mean_shap': mean_shap
        }).sort_values('mean_shap', ascending=True).head(10)
        
        print(f"\n🟢 Top Features that DECREASE Fraud Probability:")
        for idx, row in fraud_decreasing.iterrows():
            if row['mean_shap'] < 0:
                print(f"  {row['feature']:<40} {row['mean_shap']:>+10.6f}")
        
        return shap_importance_df
    
    def local_shap_analysis(self):
        """
        Perform local SHAP analysis for specific prediction cases (TP, FP, FN).
        """
        print(f"\n{'='*80}")
        print("🔍 STEP 5: LOCAL SHAP ANALYSIS (Individual Predictions)")
        print(f"{'='*80}")
        
        # Identify prediction cases
        cm = confusion_matrix(self.y_test_sample, self.y_pred_sample)
        tn, fp_idx, fn_idx, tp_idx = cm.ravel()
        
        print(f"\n📊 Confusion Matrix on Sample:")
        print(f"  True Negatives (TN):  {tn}")
        print(f"  False Positives (FP): {fp_idx}")
        print(f"  False Negatives (FN): {fn_idx}")
        print(f"  True Positives (TP):  {tp_idx}")
        
        # Find indices for each case type
        tp_indices = np.where((self.y_test_sample == 1) & (self.y_pred_sample == 1))[0]
        fp_indices = np.where((self.y_test_sample == 0) & (self.y_pred_sample == 1))[0]
        fn_indices = np.where((self.y_test_sample == 1) & (self.y_pred_sample == 0))[0]
        
        cases = {}
        
        # True Positive example
        if len(tp_indices) > 0:
            tp_example_idx = tp_indices[0]
            cases['TP'] = {
                'index': tp_example_idx,
                'y_true': self.y_test_sample[tp_example_idx],
                'y_pred': self.y_pred_sample[tp_example_idx],
                'y_proba': self.y_pred_proba_sample[tp_example_idx],
                'shap_values': self.shap_values[tp_example_idx],
                'features': self.X_test_sample[tp_example_idx]
            }
            print(f"\n✅ True Positive Example Found:")
            print(f"   Actual: Fraud (1), Predicted: Fraud (1)")
            print(f"   Probability: {cases['TP']['y_proba']:.4f}")
        
        # False Positive example
        if len(fp_indices) > 0:
            fp_example_idx = fp_indices[0]
            cases['FP'] = {
                'index': fp_example_idx,
                'y_true': self.y_test_sample[fp_example_idx],
                'y_pred': self.y_pred_sample[fp_example_idx],
                'y_proba': self.y_pred_proba_sample[fp_example_idx],
                'shap_values': self.shap_values[fp_example_idx],
                'features': self.X_test_sample[fp_example_idx]
            }
            print(f"\n⚠️  False Positive Example Found:")
            print(f"   Actual: Legitimate (0), Predicted: Fraud (1)")
            print(f"   Probability: {cases['FP']['y_proba']:.4f}")
        
        # False Negative example
        if len(fn_indices) > 0:
            fn_example_idx = fn_indices[0]
            cases['FN'] = {
                'index': fn_example_idx,
                'y_true': self.y_test_sample[fn_example_idx],
                'y_pred': self.y_pred_sample[fn_example_idx],
                'y_proba': self.y_pred_proba_sample[fn_example_idx],
                'shap_values': self.shap_values[fn_example_idx],
                'features': self.X_test_sample[fn_example_idx]
            }
            print(f"\n❌ False Negative Example Found:")
            print(f"   Actual: Fraud (1), Predicted: Legitimate (0)")
            print(f"   Probability: {cases['FN']['y_proba']:.4f}")
        
        # Generate force plots for each case
        os.makedirs('temp_charts', exist_ok=True)
        
        for case_type, case_data in cases.items():
            print(f"\n📊 Generating Force Plot for {case_type}...")
            
            # Create force plot
            force_plot = shap.force_plot(
                self.shap_values_expected,
                case_data['shap_values'],
                case_data['features'],
                feature_names=self.feature_names,
                show=False,
                matplotlib=True
            )
            
            # Save as HTML (interactive)
            html_filename = f"temp_charts/shap_force_plot_{case_type}_{self.dataset_name.lower().replace(' ', '_')}.html"
            shap.save_html(html_filename, force_plot)
            print(f"  💾 Interactive force plot saved: {html_filename}")
            
            # Analyze top contributing features
            self._analyze_case_features(case_type, case_data)
        
        return cases
    
    def _analyze_case_features(self, case_type, case_data):
        """
        Analyze top contributing features for a specific case.
        """
        # Get top contributing features (by absolute SHAP value)
        feature_contributions = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': case_data['shap_values'],
            'feature_value': case_data['features']
        }).sort_values('shap_value', key=abs, ascending=False)
        
        print(f"\n📋 Top 5 Features Contributing to {case_type} Prediction:")
        print("="*80)
        for idx, row in feature_contributions.head(5).iterrows():
            direction = "↑ Increases" if row['shap_value'] > 0 else "↓ Decreases"
            print(f"  {row['feature']:<40} {direction:>12} fraud probability")
            print(f"    SHAP Value: {row['shap_value']:>+8.4f} | Feature Value: {row['feature_value']:>10.4f}")
    
    def compare_importance_methods(self, builtin_importance_df, shap_importance_df):
        """
        Compare built-in feature importance with SHAP importance.
        """
        print(f"\n{'='*80}")
        print("📊 STEP 6: COMPARING IMPORTANCE METHODS")
        print(f"{'='*80}")
        
        # Merge importance dataframes
        comparison_df = builtin_importance_df.merge(
            shap_importance_df,
            on='feature',
            how='outer',
            suffixes=('_builtin', '_shap')
        )
        
        # Normalize importance scores (0-1 scale)
        comparison_df['importance_builtin_norm'] = (
            comparison_df['importance'] / comparison_df['importance'].max()
        )
        comparison_df['importance_shap_norm'] = (
            comparison_df['mean_abs_shap'] / comparison_df['mean_abs_shap'].max()
        )
        
        # Calculate correlation
        correlation = comparison_df['importance_builtin_norm'].corr(
            comparison_df['importance_shap_norm']
        )
        
        print(f"\n📈 Correlation between Built-in and SHAP Importance: {correlation:.4f}")
        
        if correlation > 0.7:
            print("  ✓ Strong agreement between methods")
        elif correlation > 0.4:
            print("  ⚠ Moderate agreement - some differences exist")
        else:
            print("  ⚠ Weak agreement - significant differences")
        
        # Visualize comparison
        top_features = comparison_df.nlargest(15, 'mean_abs_shap')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Built-in importance
        ax1.barh(range(len(top_features)), top_features['importance_builtin_norm'], 
                color='steelblue', alpha=0.7)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.set_xlabel('Normalized Importance', fontsize=12, fontweight='bold')
        ax1.set_title('Built-in Feature Importance\n(Top 15 by SHAP)', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # SHAP importance
        ax2.barh(range(len(top_features)), top_features['importance_shap_norm'], 
                color='darkgreen', alpha=0.7)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['feature'])
        ax2.set_xlabel('Normalized Importance', fontsize=12, fontweight='bold')
        ax2.set_title('SHAP Feature Importance\n(Top 15 by SHAP)', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        filename = f"temp_charts/importance_comparison_{self.dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n  💾 Importance comparison plot saved: {filename}")
        plt.close()
        
        return comparison_df
    
    def generate_business_recommendations(self, shap_importance_df, cases):
        """
        Generate actionable business recommendations based on SHAP insights.
        """
        print(f"\n{'='*80}")
        print("💼 STEP 7: BUSINESS RECOMMENDATIONS")
        print(f"{'='*80}")
        
        # Get top 5 fraud drivers
        top_5_drivers = shap_importance_df.head(5)
        
        # Analyze feature patterns
        mean_shap = self.shap_values.mean(axis=0)
        feature_impact_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_shap': mean_shap,
            'mean_abs_shap': np.abs(self.shap_values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        recommendations = []
        
        print(f"\n🎯 Top 5 Most Influential Fraud Drivers:")
        print("="*80)
        for idx, (_, row) in enumerate(top_5_drivers.iterrows(), 1):
            impact_direction = "increases" if feature_impact_df[
                feature_impact_df['feature'] == row['feature']
            ]['mean_shap'].values[0] > 0 else "decreases"
            
            print(f"\n  {idx}. {row['feature']}")
            print(f"     Impact: {impact_direction} fraud probability")
            print(f"     SHAP Importance: {row['mean_abs_shap']:.6f}")
        
        # Generate specific recommendations
        print(f"\n💡 Actionable Business Recommendations:")
        print("="*80)
        
        # Recommendation 1: Based on top driver
        top_feature = top_5_drivers.iloc[0]['feature']
        top_impact = feature_impact_df[feature_impact_df['feature'] == top_feature]['mean_shap'].values[0]
        
        rec1 = f"Monitor and implement threshold-based rules for '{top_feature}'"
        rec1_just = f"SHAP analysis shows '{top_feature}' has the strongest impact on fraud predictions ({top_impact:+.4f} mean SHAP value). Transactions with extreme values for this feature should trigger additional verification."
        recommendations.append({'recommendation': rec1, 'justification': rec1_just})
        print(f"\n✅ Recommendation 1:")
        print(f"   {rec1}")
        print(f"   Justification: {rec1_just}")
        
        # Recommendation 2: Feature interaction patterns
        if 'FP' in cases:
            fp_top_features = pd.DataFrame({
                'feature': self.feature_names,
                'shap_value': cases['FP']['shap_values']
            }).nlargest(3, 'shap_value', key=abs)
            
            fp_features_list = ', '.join(fp_top_features['feature'].tolist())
            rec2 = f"Review false positive patterns related to: {fp_features_list}"
            rec2_just = f"False positive analysis reveals that these features often incorrectly flag legitimate transactions. Consider adjusting thresholds or implementing multi-factor authentication to reduce false alarms while maintaining fraud detection."
            recommendations.append({'recommendation': rec2, 'justification': rec2_just})
            print(f"\n✅ Recommendation 2:")
            print(f"   {rec2}")
            print(f"   Justification: {rec2_just}")
        
        # Recommendation 3: High-risk feature combinations
        top_3_features = top_5_drivers.head(3)['feature'].tolist()
        rec3 = f"Implement multi-feature risk scoring combining: {', '.join(top_3_features)}"
        rec3_just = f"SHAP analysis identifies these as the top 3 fraud drivers. A combined risk score using multiple features (rather than single-feature rules) will improve detection accuracy and reduce both false positives and false negatives."
        recommendations.append({'recommendation': rec3, 'justification': rec3_just})
        print(f"\n✅ Recommendation 3:")
        print(f"   {rec3}")
        print(f"   Justification: {rec3_just}")
        
        # Additional recommendations based on feature patterns
        # Check for time-based features
        time_features = [f for f in self.feature_names if any(x in f.lower() for x in ['time', 'hour', 'day', 'week'])]
        if time_features:
            time_feature = [f for f in time_features if f in top_5_drivers['feature'].values]
            if time_feature:
                rec4 = f"Implement time-based fraud detection rules for {time_feature[0]}"
                rec4_just = f"SHAP analysis shows temporal patterns in fraud. Transactions occurring during specific time periods (identified by {time_feature[0]}) have higher fraud probability. Implement time-based alerts and verification."
                recommendations.append({'recommendation': rec4, 'justification': rec4_just})
                print(f"\n✅ Recommendation 4:")
                print(f"   {rec4}")
                print(f"   Justification: {rec4_just}")
        
        # Check for amount/value features
        amount_features = [f for f in self.feature_names if any(x in f.lower() for x in ['amount', 'value', 'price'])]
        if amount_features:
            amount_feature = [f for f in amount_features if f in top_5_drivers['feature'].values]
            if amount_feature:
                rec5 = f"Implement tiered verification based on {amount_feature[0]} thresholds"
                rec5_just = f"SHAP analysis reveals {amount_feature[0]} as a key fraud driver. Implement different verification levels (e.g., low/medium/high) based on transaction values, with stricter checks for higher amounts."
                recommendations.append({'recommendation': rec5, 'justification': rec5_just})
                print(f"\n✅ Recommendation 5:")
                print(f"   {rec5}")
                print(f"   Justification: {rec5_just}")
        
        print(f"\n{'='*80}")
        print("📋 SUMMARY")
        print(f"{'='*80}")
        print(f"\n✓ Identified top 5 fraud drivers using SHAP analysis")
        print(f"✓ Generated {len(recommendations)} actionable business recommendations")
        print(f"✓ Each recommendation is directly linked to SHAP insights")
        print(f"\n💼 Next Steps:")
        print(f"  1. Review recommendations with fraud prevention team")
        print(f"  2. Implement pilot programs for top recommendations")
        print(f"  3. Monitor impact on fraud detection metrics")
        print(f"  4. Continuously refine rules based on new data")
        
        return recommendations


def load_or_train_model(dataset_path, target_col, dataset_name, model_type='xgboost'):
    """
    Load a saved model or train a new one if not found.
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset CSV
    target_col : str
        Target column name
    dataset_name : str
        Name of dataset
    model_type : str
        Type of model to train ('xgboost' or 'random_forest')
    """
    # Check if model exists
    model_filename = f"models/{dataset_name.lower().replace(' ', '_')}_{model_type.title()}.pkl"
    
    if os.path.exists(model_filename):
        print(f"✓ Loading saved model: {model_filename}")
        model = joblib.load(model_filename)
        return model, None, None, None, None
    
    # If model doesn't exist, train one
    print(f"⚠ Model not found. Training new {model_type} model...")
    # Import trainer module
    sys.path.insert(0, os.path.dirname(__file__))
    from train_fraud_models import FraudDetectionModelTrainer
    
    trainer = FraudDetectionModelTrainer(
        dataset_path=dataset_path,
        target_col=target_col,
        dataset_name=dataset_name,
        random_state=42
    )
    
    # Prepare data
    trainer.stratified_train_test_split(test_size=0.2)
    
    # Train ensemble model
    model, _ = trainer.train_ensemble_model(model_type=model_type)
    
    # Save model
    trainer.save_model(model, model_type.title())
    
    return model, trainer.X_train, trainer.X_test, trainer.y_test, trainer.scaler


def main():
    """
    Main function to run SHAP explainability analysis.
    """
    print("\n" + "="*80)
    print("🚀 SHAP MODEL EXPLAINABILITY ANALYSIS")
    print("="*80)
    
    # Dataset configurations
    datasets = [
        {
            'path': 'data/raw/Fraud_Data.csv',
            'target': 'class',
            'name': 'E-commerce Fraud',
            'model_type': 'xgboost'
        },
        {
            'path': 'data/raw/creditcard.csv',
            'target': 'Class',
            'name': 'Credit Card Fraud',
            'model_type': 'xgboost'
        }
    ]
    
    # Process each dataset
    for dataset_config in datasets:
        try:
            print(f"\n{'='*80}")
            print(f"📊 PROCESSING: {dataset_config['name']}")
            print(f"{'='*80}")
            
            # Load or train model
            model, X_train, X_test, y_test, scaler = load_or_train_model(
                dataset_path=dataset_config['path'],
                target_col=dataset_config['target'],
                dataset_name=dataset_config['name'],
                model_type=dataset_config['model_type']
            )
            
            # If model was loaded, need to prepare data
            if X_train is None:
                print("\n📊 Preparing data for SHAP analysis...")
                df = load_data(dataset_config['path'])
                
                # Separate features and target
                X = df.drop(columns=[dataset_config['target']])
                y = df[dataset_config['target']]
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train = pd.DataFrame(
                    scaler.fit_transform(X_train),
                    columns=X_train.columns,
                    index=X_train.index
                )
                X_test = pd.DataFrame(
                    scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
            
            # Initialize SHAP explainer
            explainer = SHAPFraudExplainer(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=list(X_train.columns),
                dataset_name=dataset_config['name']
            )
            
            # Step 1: Initialize SHAP explainer
            explainer.initialize_shap_explainer(sample_size=500)
            
            # Step 2: Compute SHAP values
            explainer.compute_shap_values(max_samples=500)
            
            # Step 3: Built-in feature importance
            builtin_importance = explainer.analyze_builtin_feature_importance()
            
            # Step 4: Global SHAP analysis
            shap_importance = explainer.global_shap_analysis()
            
            # Step 5: Local SHAP analysis
            cases = explainer.local_shap_analysis()
            
            # Step 6: Compare importance methods
            if builtin_importance is not None:
                comparison = explainer.compare_importance_methods(builtin_importance, shap_importance)
            
            # Step 7: Business recommendations
            recommendations = explainer.generate_business_recommendations(shap_importance, cases)
            
            print(f"\n{'='*80}")
            print(f"✅ COMPLETE: {dataset_config['name']}")
            print(f"{'='*80}")
            print(f"\n📁 Output Files Generated:")
            print(f"  • Feature importance plots: temp_charts/")
            print(f"  • SHAP summary plots: temp_charts/")
            print(f"  • Force plots (interactive HTML): temp_charts/")
            print(f"  • Comparison visualizations: temp_charts/")
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: Dataset not found - {dataset_config['path']}")
            continue
        except Exception as e:
            print(f"\n❌ Error processing {dataset_config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*80)
    print("🎉 SHAP ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

