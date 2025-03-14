"""
ml_model.py - Machine Learning model script for Patient Readmission Analysis project

This script builds, trains, and evaluates machine learning models to predict
the likelihood of patient readmission using various algorithms.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import argparse
from typing import Dict, List, Optional, Union, Tuple
import joblib

# Machine learning imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ml_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('output/model_evaluation', exist_ok=True)


class ReadmissionPredictor:
    """Class to build and evaluate machine learning models for readmission prediction"""

    def __init__(self, data_path: Optional[str] = None, model_dir: str = 'models'):
        """Initialize the predictor with data path and model directory"""
        self.data_path = data_path
        self.model_dir = model_dir
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        self.target_column = None
        self.preprocessor = None

        os.makedirs(model_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV or database"""
        if self.data_path and os.path.exists(self.data_path):
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
        else:
            processed_dir = 'data/processed'
            if os.path.exists(processed_dir):
                files = [f for f in os.listdir(processed_dir) if f.startswith('ml_features_dataset')]
                if files:
                    latest_file = sorted(files)[-1]
                    file_path = os.path.join(processed_dir, latest_file)
                    logger.info(f"Loading data from {file_path}")
                    self.df = pd.read_csv(file_path)
                else:
                    raise FileNotFoundError("No processed data files found")
            else:
                raise FileNotFoundError("No data found. Please provide a data path or run the ETL pipeline")
        
        logger.info(f"Loaded dataset with {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df

    def identify_column_types(self) -> Tuple[List[str], List[str], str]:
        """Identify numerical and categorical columns and the target column"""
        target_columns = [col for col in self.df.columns if 'readmi' in col.lower()]
        for col in ['readmitted_30_days', 'is_30day_readmission', 'is_readmission']:
            if col in self.df.columns:
                self.target_column = col
                break
        
        if not self.target_column and target_columns:
            self.target_column = target_columns[0]
            
        if not self.target_column:
            raise ValueError("No readmission target column found in the data")
            
        logger.info(f"Using '{self.target_column}' as the target column")

        exclude_cols = [self.target_column, 'patient_id', 'stay_id', 'admission_date', 
                        'discharge_date', 'previous_admission_date', 'previous_discharge_date']
        
        date_cols = [col for col in self.df.columns if 'date' in col.lower()]
        exclude_cols.extend(date_cols)
        
        exclude_cols = [col for col in exclude_cols if col in self.df.columns]
        
        self.categorical_features = self.df.select_dtypes(include=['object', 'bool']).columns.tolist()
        self.categorical_features = [col for col in self.categorical_features if col not in exclude_cols]
        
        self.numerical_features = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.numerical_features = [col for col in self.numerical_features if col not in exclude_cols]
        
        logger.info(f"Identified {len(self.categorical_features)} categorical features and {len(self.numerical_features)} numerical features")
        
        return self.numerical_features, self.categorical_features, self.target_column

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess data for model training"""
        logger.info("Preprocessing data for model training")
        
        if not self.numerical_features or not self.target_column:
            self.identify_column_types()
        
        X = self.df[self.numerical_features + self.categorical_features]
        y = self.df[self.target_column].astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        joblib.dump(self.preprocessor, os.path.join(self.model_dir, 'preprocessor.joblib'))
        
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        try:
            cat_features = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
            self.feature_names = np.concatenate([self.numerical_features, cat_features])
        except:
            logger.warning("Could not extract feature names after preprocessing")
            self.feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
        
        logger.info(f"Preprocessed training data shape: {X_train_processed.shape}")
        logger.info(f"Preprocessed test data shape: {X_test_processed.shape}")
        
        return X_train_processed, X_test_processed, y_train, y_test

    def build_models(self) -> Dict[str, Pipeline]:
        """Build various classification models"""
        logger.info("Building classification models")
        
        self.models = {
            'logistic_regression': Pipeline([
                ('model', LogisticRegression(max_iter=1000, random_state=42))
            ]),
            'decision_tree': Pipeline([
                ('model', DecisionTreeClassifier(random_state=42))
            ]),
            'random_forest': Pipeline([
                ('model', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'gradient_boosting': Pipeline([
                ('model', GradientBoostingClassifier(n_estimators=100, random_state=42))
            ]),
            'k_neighbors': Pipeline([
                ('model', KNeighborsClassifier(n_neighbors=5))
            ]),
            'svm': Pipeline([
                ('model', CalibratedClassifierCV(SVC(probability=True, random_state=42)))
            ]),
            'xgboost': Pipeline([
                ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
            ])
        }
        
        return self.models

    def train_evaluate_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                             y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """Train and evaluate all models"""
        logger.info("Training and evaluating models")
        
        if not self.models:
            self.build_models()
        
        results = {}
        
        for name, pipeline in self.models.items():
            logger.info(f"Training {name} model")
            
            try:
                pipeline.fit(X_train, y_train)
                
                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_prob)
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
                results[name]['cv_roc_auc_mean'] = cv_scores.mean()
                results[name]['cv_roc_auc_std'] = cv_scores.std()
                
                joblib.dump(pipeline, os.path.join(self.model_dir, f"{name}_model.joblib"))
                
                logger.info(f"{name} model: ROC AUC = {roc_auc:.4f}, F1 = {f1:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name} model: {str(e)}")
                results[name] = {'error': str(e)}
        
        self.results = results
        
        best_model = max(results.items(), key=lambda x: x[1]['roc_auc'] if 'roc_auc' in x[1] else 0)
        logger.info(f"Best model: {best_model[0]} with ROC AUC = {best_model[1]['roc_auc']:.4f}")
        
        return results

    def visualize_results(self) -> None:
        """Create visualizations of model performance"""
        logger.info("Creating model performance visualizations")
        
        if not self.results:
            logger.warning("No model results to visualize")
            return
        
        output_dir = 'output/model_evaluation'
        os.makedirs(output_dir, exist_ok=True)
        
        valid_models = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_models:
            logger.warning("No valid model results to visualize")
            return
        
        # 1. ROC Curves
        plt.figure(figsize=(10, 8))
        
        for name, result in valid_models.items():
            if 'roc_auc' in result:
                model = joblib.load(os.path.join(self.model_dir, f"{name}_model.joblib"))
                
                # Get predictions
                y_prob = model.predict_proba(self.X_test)[:, 1]
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                
                plt.plot(fpr, tpr, label=f"{name} (AUC = {result['roc_auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Readmission Prediction Models')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model Comparison (bar chart of key metrics)
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        metric_data = []
        
        for name, result in valid_models.items():
            for metric in metrics:
                if metric in result:
                    metric_data.append({
                        'Model': name,
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': result[metric]
                    })
        
        if metric_data:
            metric_df = pd.DataFrame(metric_data)
            
            plt.figure(figsize=(14, 10))
            sns.barplot(x='Model', y='Value', hue='Metric', data=metric_df)
            plt.title('Comparison of Model Performance Metrics')
            plt.xlabel('Model')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.legend(title='Metric')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Feature Importance (for models that support it)
        for name, result in valid_models.items():
            try:
                model = joblib.load(os.path.join(self.model_dir, f"{name}_model.joblib"))
                
                # Check if model has feature_importances_ attribute
                if hasattr(model['model'], 'feature_importances_'):
                    # Get feature importances
                    importances = model['model'].feature_importances_
                    
                    if len(importances) == len(self.feature_names):
                        importance_df = pd.DataFrame({
                            'Feature': self.feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False).head(20)
                        
                        plt.figure(figsize=(12, 8))
                        sns.barplot(x='Importance', y='Feature', data=importance_df)
                        plt.title(f'Top 20 Feature Importances ({name})')
                        plt.xlabel('Importance')
                        plt.ylabel('Feature')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f'{name}_feature_importance.png'), dpi=300, bbox_inches='tight')
                        plt.close()
                elif name == 'logistic_regression':
                    # For logistic regression, use coefficients
                    coefficients = model['model'].coef_[0]
                    
                    if len(coefficients) == len(self.feature_names):
                        coef_df = pd.DataFrame({
                            'Feature': self.feature_names,
                            'Coefficient': coefficients
                        }).sort_values('Coefficient', key=abs, ascending=False).head(20)
                        
                        plt.figure(figsize=(12, 8))
                        sns.barplot(x='Coefficient', y='Feature', data=coef_df)
                        plt.title(f'Top 20 Feature Coefficients (Logistic Regression)')
                        plt.xlabel('Coefficient')
                        plt.ylabel('Feature')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, 'logistic_regression_coefficients.png'), dpi=300, bbox_inches='tight')
                        plt.close()
            except Exception as e:
                logger.warning(f"Could not create feature importance plot for {name}: {str(e)}")
        
        # 4. Confusion Matrices
        for name, result in valid_models.items():
            if 'confusion_matrix' in result:
                cm = np.array(result['confusion_matrix'])
                
                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Not Readmitted', 'Readmitted'],
                           yticklabels=['Not Readmitted', 'Readmitted'])
                plt.title(f'Confusion Matrix ({name})')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        # 5. Cross-validation results
        cv_data = []
        for name, result in valid_models.items():
            if 'cv_roc_auc_mean' in result and 'cv_roc_auc_std' in result:
                cv_data.append({
                    'Model': name,
                    'CV ROC AUC': result['cv_roc_auc_mean'],
                    'Std Dev': result['cv_roc_auc_std']
                })
        
        if cv_data:
            cv_df = pd.DataFrame(cv_data)
            
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Model', y='CV ROC AUC', data=cv_df)
            
            for i, row in cv_df.iterrows():
                ax.errorbar(i, row['CV ROC AUC'], yerr=row['Std Dev'], fmt='o', color='black')
            
            plt.title('Cross-Validation ROC AUC Scores (5-fold)')
            plt.xlabel('Model')
            plt.ylabel('ROC AUC')
            plt.ylim(0.5, 1.0)
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cross_validation_results.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Model evaluation visualizations saved to {output_dir}")

    def hyperparameter_tuning(self, best_model_name: Optional[str] = None) -> Dict:
        """Perform hyperparameter tuning for the best model"""
        logger.info("Performing hyperparameter tuning")
        
        if not best_model_name:
            valid_models = {k: v for k, v in self.results.items() if 'error' not in v and 'roc_auc' in v}
            if not valid_models:
                logger.warning("No valid models found for hyperparameter tuning")
                return {}
            
            best_model_name = max(valid_models.items(), key=lambda x: x[1]['roc_auc'])[0]
        
        logger.info(f"Tuning hyperparameters for {best_model_name} model")
        
        X_train_processed = self.preprocessor.transform(self.X_train)
        X_test_processed = self.preprocessor.transform(self.X_test)
        
        # Define hyperparameter grids for different models
        param_grids = {
            'logistic_regression': {
                'model__C': [0.01, 0.1, 1, 10, 100],
                'model__penalty': ['l1', 'l2'],
                'model__solver': ['liblinear', 'saga']
            },
            'decision_tree': {
                'model__max_depth': [3, 5, 7, 10, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            },
            'random_forest': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [5, 10, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            },
            'k_neighbors': {
                'model__n_neighbors': [3, 5, 7, 9, 11],
                'model__weights': ['uniform', 'distance'],
                'model__metric': ['euclidean', 'manhattan']
            },
            'svm': {
                'model__estimator__C': [0.1, 1, 10],
                'model__estimator__kernel': ['linear', 'rbf'],
                'model__estimator__gamma': ['scale', 'auto']
            },
            'xgboost': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__subsample': [0.8, 0.9, 1.0]
            }
        }
        
        # Check if the best model has a param grid
        if best_model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {best_model_name}")
            return {}
        
        model = self.models[best_model_name]
        
        # Create grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[best_model_name],
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_processed, self.y_train)
        
        # Get best parameters
        best_params = grid_search.best_params_
        logger.info(f"Best parameters for {best_model_name}: {best_params}")
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_processed)
        y_prob = best_model.predict_proba(X_test_processed)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_prob)
        
        # Store results
        tuning_results = {
            'best_params': best_params,
            'best_score': grid_search.best_score_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
        }
        
        # Compare with the original model
        original_roc_auc = self.results[best_model_name]['roc_auc']
        improvement = (roc_auc - original_roc_auc) / original_roc_auc * 100
        
        logger.info(f"Tuned {best_model_name} model: ROC AUC = {roc_auc:.4f} (improvement: {improvement:.2f}%)")
        
        # Save the tuned model
        joblib.dump(best_model, os.path.join(self.model_dir, f"{best_model_name}_tuned_model.joblib"))
        
        # Create comparison visualization
        self._visualize_tuning_results(best_model_name, tuning_results)
        
        return tuning_results

    def _visualize_tuning_results(self, model_name: str, tuning_results: Dict) -> None:
        """Visualize the results of hyperparameter tuning"""
        output_dir = 'output/model_evaluation'
        os.makedirs(output_dir, exist_ok=True)
        
        # Compare original vs. tuned model metrics
        original_metrics = self.results[model_name]
        tuned_metrics = tuning_results
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        comparison_data = []
        
        for metric in metrics:
            comparison_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Original': original_metrics[metric],
                'Tuned': tuned_metrics[metric]
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Reshape for plotting
        comparison_melted = pd.melt(
            comparison_df, 
            id_vars=['Metric'], 
            value_vars=['Original', 'Tuned'],
            var_name='Model Version', 
            value_name='Score'
        )
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Metric', y='Score', hue='Model Version', data=comparison_melted)
        plt.title(f'Original vs. Tuned {model_name.replace("_", " ").title()} Model')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.legend(title='Model Version')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_tuning_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        
        # Original model
        original_model = joblib.load(os.path.join(self.model_dir, f"{model_name}_model.joblib"))
        original_probs = original_model.predict_proba(self.X_test)[:, 1]
        original_fpr, original_tpr, _ = roc_curve(self.y_test, original_probs)
        plt.plot(original_fpr, original_tpr, label=f"Original (AUC = {original_metrics['roc_auc']:.3f})")
        
        # Tuned model
        tuned_model = joblib.load(os.path.join(self.model_dir, f"{model_name}_tuned_model.joblib"))
        tuned_probs = tuned_model.predict_proba(self.X_test)[:, 1]
        tuned_fpr, tuned_tpr, _ = roc_curve(self.y_test, tuned_probs)
        plt.plot(tuned_fpr, tuned_tpr, label=f"Tuned (AUC = {tuned_metrics['roc_auc']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves: Original vs. Tuned {model_name.replace("_", " ").title()} Model')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'{model_name}_tuning_roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def run_pipeline(self) -> Dict:
        """Run the full machine learning pipeline"""
        logger.info("Running full machine learning pipeline")
        
        # 1. Load data
        self.load_data()
        
        # 2. Identify column types
        self.identify_column_types()
        
        # 3. Preprocess data
        X_train_processed, X_test_processed, y_train, y_test = self.preprocess_data()
        
        # 4. Build models
        self.build_models()
        
        # 5. Train and evaluate models
        results = self.train_evaluate_models(X_train_processed, X_test_processed, y_train, y_test)
        
        # 6. Visualize results
        self.visualize_results()
        
        # 7. Perform hyperparameter tuning on the best model
        best_model_name = max(
            {k: v for k, v in results.items() if 'error' not in v and 'roc_auc' in v},
            key=lambda k: results[k]['roc_auc']
        )
        tuning_results = self.hyperparameter_tuning(best_model_name)
        
        # 8. Save final results
        final_results = {
            'models': results,
            'best_model': best_model_name,
            'hyperparameter_tuning': tuning_results
        }
        
        # Save results to file
        results_path = os.path.join(self.model_dir, 'model_results.json')
        import json
        
        # Convert numpy values to Python natives
        def convert_numpy(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj
        
        with open(results_path, 'w') as f:
            json.dump(convert_numpy(final_results), f, indent=2)
        
        logger.info(f"Machine learning pipeline complete. Results saved to {results_path}")
        
        return final_results


def main():
    """Main function to run the machine learning modeling process"""
    parser = argparse.ArgumentParser(description='Build and evaluate readmission prediction models')
    parser.add_argument('--data-path', help='Path to the input data CSV file')
    parser.add_argument('--model-dir', default='models', help='Directory to save models')
    args = parser.parse_args()
    
    try:
        predictor = ReadmissionPredictor(data_path=args.data_path, model_dir=args.model_dir)
        
        predictor.run_pipeline()
        
        return 0
    except Exception as e:
        logger.error(f"Error in machine learning pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())