"""
analysis.py - Data analysis and visualization script for Patient Readmission Analysis project

This script performs various analyses on the hospital readmission data, including:
- Descriptive statistics on readmission rates
- Predictive modeling to identify factors contributing to readmissions
- Visualization of key findings
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import mysql.connector
from mysql.connector import Error
import json
import logging
from datetime import datetime
import argparse
from typing import Dict, List, Optional, Union, Tuple
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    silhouette_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from joblib import dump

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.makedirs('logs', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('output/figures', exist_ok=True)
os.makedirs('output/models', exist_ok=True)
os.makedirs('output/reports', exist_ok=True)


class DatabaseConnector:
    """Class to handle database connections and queries"""
    
    def __init__(self, config_file: str = 'config/database.yaml'):
        """Initialize with database configuration"""
        self.config_file = config_file
        self.connection = None
        self.cursor = None
        
        self.config = self._load_config()
        
        self._connect()
    
    def _load_config(self) -> Dict:
        """Load database configuration from YAML file"""
        default_config = {
            'host': 'localhost',
            'database': 'healthcare_readmission',
            'user': 'root',
            'password': '',
            'port': 3306
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    return config
            else:
                logger.warning(f"Config file {self.config_file} not found, using default")
                return default_config
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            return default_config
    
    def _connect(self) -> None:
        """Connect to the MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=self.config.get('host', 'localhost'),
                database=self.config.get('database', 'healthcare_readmission'),
                user=self.config.get('user', 'root'),
                password=self.config.get('password', ''),
                port=self.config.get('port', 3306)
            )
            
            if self.connection.is_connected():
                self.cursor = self.connection.cursor(dictionary=True)
                db_info = self.connection.get_server_info()
                logger.info(f"Connected to MySQL Server version {db_info}")
        except Error as e:
            logger.error(f"Error connecting to MySQL: {str(e)}")
            raise
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> Optional[pd.DataFrame]:
        """Execute a SQL query and return results as a DataFrame"""
        try:
            self.cursor.execute(query, params or ())
            result = self.cursor.fetchall()
            return pd.DataFrame(result)
        except Error as e:
            logger.error(f"Error executing query: {str(e)}")
            return None
    
    def close(self) -> None:
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            if self.cursor:
                self.cursor.close()
            self.connection.close()
            logger.info("Database connection closed")


class ReadmissionAnalyzer:
    """Class to analyze patient readmission data"""
    
    def __init__(self, db_connector: DatabaseConnector, output_dir: str = 'output'):
        """Initialize with database connector and output directory"""
        self.db = db_connector
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, 'figures')
        self.models_dir = os.path.join(output_dir, 'models')
        self.reports_dir = os.path.join(output_dir, 'reports')
        
        # Ensure output directories exist
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # For storing analysis results
        self.results = {}
    
    def get_readmission_data(self) -> pd.DataFrame:
        """Get readmission data from database"""
        query = """
        SELECT * FROM readmission_analysis
        """
        df = self.db.execute_query(query)
        if df is None or df.empty:
            logger.warning("No readmission data found in database")
            processed_dir = 'data/processed'
            files = [f for f in os.listdir(processed_dir) if f.startswith('ml_features_dataset')]
            if files:
                latest_file = sorted(files)[-1]
                file_path = os.path.join(processed_dir, latest_file)
                logger.info(f"Loading readmission data from {file_path}")
                df = pd.read_csv(file_path)
            else:
                raise ValueError("No readmission data available")
        
        logger.info(f"Loaded readmission data with {len(df)} rows and {len(df.columns)} columns")
        return df

    def prepare_data_for_modeling(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        """Prepare data for predictive modeling"""
        if 'readmitted_30_days' in df.columns:
            target = 'readmitted_30_days'
        elif 'is_30day_readmission' in df.columns:
            target = 'is_30day_readmission'
        elif 'is_readmission' in df.columns:
            target = 'is_readmission'
        else:
            raise ValueError("No readmission target column found in data")
        
        y = df[target]
        
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        exclude_cols = [target, 'patient_id', 'stay_id', 'admission_date', 'discharge_date']
        num_features = [col for col in num_cols if col not in exclude_cols]
        cat_features = [col for col in cat_cols if col not in exclude_cols]
        
        X = df[num_features + cat_features]
        
        logger.info(f"Prepared data with {len(X)} samples, {len(num_features)} numerical features, and {len(cat_features)} categorical features")
        
        return X, y, num_features, cat_features
    
    def descriptive_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate descriptive statistics on readmission data"""
        logger.info("Calculating descriptive statistics")
        
        if 'readmitted_30_days' in df.columns:
            readmission_col = 'readmitted_30_days'
        elif 'is_30day_readmission' in df.columns:
            readmission_col = 'is_30day_readmission'
        elif 'is_readmission' in df.columns:
            readmission_col = 'is_readmission'
        else:
            logger.warning("No readmission column found, cannot calculate readmission statistics")
            return {}
        
        overall_rate = df[readmission_col].mean() * 100
        
        results = {
            'overall_readmission_rate': overall_rate,
            'total_patients': df['patient_id'].nunique(),
            'total_stays': len(df),
            'readmissions_count': df[readmission_col].sum()
        }
        
        if 'age' in df.columns:
            age_groups = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 80, 120], labels=['0-18', '19-35', '36-50', '51-65', '66-80', '81+'])
            age_rates = df.groupby(age_groups)[readmission_col].mean() * 100
            results['readmission_by_age_group'] = age_rates.to_dict()
        
        if 'gender' in df.columns:
            gender_rates = df.groupby('gender')[readmission_col].mean() * 100
            results['readmission_by_gender'] = gender_rates.to_dict()
        
        if 'race' in df.columns:
            race_rates = df.groupby('race')[readmission_col].mean() * 100
            results['readmission_by_race'] = race_rates.to_dict()
        
        if 'length_of_stay' in df.columns:
            los_bins = pd.cut(df['length_of_stay'], bins=[0, 1, 3, 7, 14, 30, 100], labels=['1 day', '2-3 days', '4-7 days', '8-14 days', '15-30 days', '30+ days'])
            los_rates = df.groupby(los_bins)[readmission_col].mean() * 100
            results['readmission_by_length_of_stay'] = los_rates.to_dict()
        
        self.results['descriptive_statistics'] = results
        
        return results
    
    def visualize_readmission_rates(self, df: pd.DataFrame) -> None:
        """Create visualizations of readmission rates"""
        logger.info("Creating readmission rate visualizations")
        
        if 'readmitted_30_days' in df.columns:
            readmission_col = 'readmitted_30_days'
        elif 'is_30day_readmission' in df.columns:
            readmission_col = 'is_30day_readmission'
        elif 'is_readmission' in df.columns:
            readmission_col = 'is_readmission'
        else:
            logger.warning("No readmission column found, cannot visualize readmission rates")
            return
        
        plt.figure(figsize=(8, 6))
        overall_rate = df[readmission_col].mean() * 100
        plt.bar(['Non-readmitted', 'Readmitted'], [100 - overall_rate, overall_rate], color=['#2ecc71', '#e74c3c'])
        plt.title('Overall 30-Day Readmission Rate', fontsize=14)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.ylim(0, 100)
        for i, rate in enumerate([100 - overall_rate, overall_rate]):
            plt.text(i, rate + 3, f'{rate:.1f}%', ha='center', fontsize=12)
        plt.savefig(os.path.join(self.figures_dir, 'overall_readmission_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        if 'age' in df.columns:
            plt.figure(figsize=(10, 6))
            age_groups = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 80, 120], labels=['0-18', '19-35', '36-50', '51-65', '66-80', '81+'])
            age_rates = df.groupby(age_groups)[readmission_col].mean() * 100
            
            sns.barplot(x=age_rates.index, y=age_rates.values, palette='viridis')
            plt.title('Readmission Rate by Age Group', fontsize=14)
            plt.xlabel('Age Group', fontsize=12)
            plt.ylabel('Readmission Rate (%)', fontsize=12)
            plt.ylim(0, age_rates.max() * 1.2)
            
            for i, rate in enumerate(age_rates):
                plt.text(i, rate + 1, f'{rate:.1f}%', ha='center', fontsize=10)
            
            plt.savefig(os.path.join(self.figures_dir, 'readmission_by_age.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        if 'length_of_stay' in df.columns:
            plt.figure(figsize=(12, 6))
            los_bins = pd.cut(df['length_of_stay'], bins=[0, 1, 3, 7, 14, 30, 100], labels=['1 day', '2-3 days', '4-7 days', '8-14 days', '15-30 days', '30+ days'])
            los_rates = df.groupby(los_bins)[readmission_col].mean() * 100
            
            sns.barplot(x=los_rates.index, y=los_rates.values, palette='viridis')
            plt.title('Readmission Rate by Length of Stay', fontsize=14)
            plt.xlabel('Length of Stay', fontsize=12)
            plt.ylabel('Readmission Rate (%)', fontsize=12)
            plt.ylim(0, los_rates.max() * 1.2)
            plt.xticks(rotation=45)
            
            for i, rate in enumerate(los_rates):
                plt.text(i, rate + 1, f'{rate:.1f}%', ha='center', fontsize=10)
            
            plt.savefig(os.path.join(self.figures_dir, 'readmission_by_length_of_stay.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        if 'primary_diagnosis' in df.columns:
            plt.figure(figsize=(14, 8))
            top_diagnoses = df['primary_diagnosis'].value_counts().head(10).index
            diagnosis_rates = df[df['primary_diagnosis'].isin(top_diagnoses)].groupby('primary_diagnosis')[readmission_col].mean() * 100
            diagnosis_rates = diagnosis_rates.sort_values(ascending=False)
            
            sns.barplot(x=diagnosis_rates.index, y=diagnosis_rates.values, palette='viridis')
            plt.title('Readmission Rate by Primary Diagnosis (Top 10)', fontsize=14)
            plt.xlabel('Primary Diagnosis', fontsize=12)
            plt.ylabel('Readmission Rate (%)', fontsize=12)
            plt.ylim(0, diagnosis_rates.max() * 1.2)
            plt.xticks(rotation=45, ha='right')
            
            for i, rate in enumerate(diagnosis_rates):
                plt.text(i, rate + 1, f'{rate:.1f}%', ha='center', fontsize=10)
            
            plt.savefig(os.path.join(self.figures_dir, 'readmission_by_diagnosis.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(num_cols) > 1:
            plt.figure(figsize=(12, 10))
            corr_matrix = df[num_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Matrix of Numerical Variables', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Saved readmission visualizations to {self.figures_dir}")
    
    def build_predictive_model(self, df: pd.DataFrame) -> Dict:
        """Build a predictive model for readmission risk"""
        logger.info("Building predictive model for readmission risk")
        
        try:
            X, y, num_features, cat_features = self.prepare_data_for_modeling(df)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), num_features),
                    ('cat', Pipeline([
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ]), cat_features)
                ]
            )
            
            models = {
                'logistic_regression': Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
                ]),
                'decision_tree': Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', DecisionTreeClassifier(random_state=42))
                ]),
                'random_forest': Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
                ])
            }
            
            results = {}
            best_model = None
            best_score = 0
            
            for name, model in models.items():
                logger.info(f"Training {name} model")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                
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
                
                model_path = os.path.join(self.models_dir, f"{name}_model.joblib")
                dump(model, model_path)
                logger.info(f"Saved {name} model to {model_path}")
                
                if roc_auc > best_score:
                    best_score = roc_auc
                    best_model = name
            
            plt.figure(figsize=(10, 8))
            for name, model in models.items():
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = stats.metrics.roc_curve(y_test, y_prob)
                roc_auc = stats.metrics.auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.figures_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            if best_model == 'random_forest':
                model = models[best_model]
                features = []
                for name, trans, cols in preprocessor.transformers_:
                    if name == 'num':
                        features.extend(cols)
                    elif name == 'cat':
                        pipe = trans.steps[1][1]  # Get the OneHotEncoder
                        cats = pipe.get_feature_names_out(cols)
                        features.extend(cats)
                
                importances = model.named_steps['classifier'].feature_importances_
                
                if len(features) == len(importances):
                    feature_importance = pd.DataFrame({
                        'feature': features,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    # Visualize top 15 features
                    plt.figure(figsize=(12, 8))
                    sns.barplot(x='importance', y='feature', data=feature_importance.head(15), palette='viridis')
                    plt.title('Top 15 Features by Importance (Random Forest)', fontsize=14)
                    plt.xlabel('Importance', fontsize=12)
                    plt.ylabel('Feature', fontsize=12)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.figures_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
                    plt.close()
            
            self.results['predictive_modeling'] = {
                'models': results,
                'best_model': best_model,
                'best_score': best_score
            }
            
            return self.results['predictive_modeling']
            
        except Exception as e:
            logger.error(f"Error building predictive model: {str(e)}")
            return {'error': str(e)}
    
    def perform_clustering(self, df: pd.DataFrame) -> Dict:
        """Perform clustering to identify patient segments"""
        logger.info("Performing clustering analysis")
        
        try:
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            exclude_cols = ['patient_id', 'stay_id']
            cluster_features = [col for col in num_cols if col not in exclude_cols
                               and 'date' not in col.lower() and 'readmi' not in col.lower()]
            
            if len(cluster_features) < 2:
                logger.warning("Not enough numerical features for clustering")
                return {'error': 'Not enough numerical features for clustering'}
            
            X_cluster = df[cluster_features].copy()
            X_cluster = X_cluster.fillna(X_cluster.median())
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            
            # Determine optimal number of clusters using silhouette score
            sil_scores = []
            k_range = range(2, min(11, len(df) // 10))  # Try 2-10 clusters, but no more than 1/10 of samples
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                sil_score = silhouette_score(X_scaled, clusters)
                sil_scores.append(sil_score)
            
            # Find optimal k (number of clusters)
            optimal_k = k_range[np.argmax(sil_scores)]
            
            # Visualize silhouette scores
            plt.figure(figsize=(10, 6))
            plt.plot(list(k_range), sil_scores, marker='o')
            plt.axvline(x=optimal_k, color='red', linestyle='--')
            plt.title('Silhouette Score by Number of Clusters', fontsize=14)
            plt.xlabel('Number of Clusters (k)', fontsize=12)
            plt.ylabel('Silhouette Score', fontsize=12)
            plt.grid(True)
            plt.savefig(os.path.join(self.figures_dir, 'silhouette_scores.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Perform final clustering with optimal k
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            df_clusters = df.copy()
            df_clusters['cluster'] = clusters
            
            # Analyze clusters
            cluster_profiles = df_clusters.groupby('cluster').agg({
                **{col: 'mean' for col in num_cols if col in df_clusters.columns},
                'patient_id': 'nunique'
            }).rename(columns={'patient_id': 'patient_count'})
            
            readmission_col = next((col for col in ['readmitted_30_days', 'is_30day_readmission', 'is_readmission'] 
                                   if col in df_clusters.columns), None)
            
            if readmission_col:
                cluster_readmission = df_clusters.groupby('cluster')[readmission_col].mean() * 100
                cluster_profiles['readmission_rate'] = cluster_readmission
            
            if len(cluster_features) > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, label='Cluster')
                plt.title('Patient Clusters (PCA Projection)', fontsize=14)
                plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
                plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(self.figures_dir, 'patient_clusters.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            if readmission_col:
                plt.figure(figsize=(10, 6))
                sns.barplot(x=cluster_readmission.index, y=cluster_readmission.values, palette='viridis')
                plt.title('Readmission Rate by Patient Cluster', fontsize=14)
                plt.xlabel('Cluster', fontsize=12)
                plt.ylabel('Readmission Rate (%)', fontsize=12)
                plt.ylim(0, cluster_readmission.max() * 1.2)
                
                for i, rate in enumerate(cluster_readmission):
                    plt.text(i, rate + 1, f'{rate:.1f}%', ha='center', fontsize=10)
                
                plt.savefig(os.path.join(self.figures_dir, 'readmission_by_cluster.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            self.results['clustering'] = {
                'optimal_clusters': optimal_k,
                'silhouette_scores': {k: score for k, score in zip(k_range, sil_scores)},
                'cluster_profiles': cluster_profiles.to_dict(),
                'features_used': cluster_features
            }
            
            cluster_model = {
                'kmeans': kmeans,
                'scaler': scaler,
                'features': cluster_features
            }
            dump(cluster_model, os.path.join(self.models_dir, 'cluster_model.joblib'))
            
            return self.results['clustering']
            
        except Exception as e:
            logger.error(f"Error performing clustering: {str(e)}")
            return {'error': str(e)}
    
    def save_analysis_report(self) -> None:
        """Save analysis results to a JSON report"""
        report_path = os.path.join(self.reports_dir, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
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
        
        processed_results = convert_numpy(self.results)
        
        with open(report_path, 'w') as f:
            json.dump(processed_results, f, indent=2)
        
        logger.info(f"Saved analysis report to {report_path}")


def main():
    """Main function to run the analysis process"""
    parser = argparse.ArgumentParser(description='Analyze patient readmission data')
    parser.add_argument('--config', default='config/database.yaml', help='Path to database config file')
    parser.add_argument('--output-dir', default='output', help='Directory to save output files')
    parser.add_argument('--use-file', help='Use processed data file instead of database')
    args = parser.parse_args()
    
    try:
        db_connector = DatabaseConnector(config_file=args.config)
        
        analyzer = ReadmissionAnalyzer(db_connector, output_dir=args.output_dir)
        
        if args.use_file and os.path.exists(args.use_file):
            logger.info(f"Loading data from file: {args.use_file}")
            df = pd.read_csv(args.use_file)
        else:
            logger.info("Loading data from database")
            df = analyzer.get_readmission_data()
        
        if df.empty:
            logger.error("No data available for analysis")
            return 1
        
        analyzer.descriptive_statistics(df)
        analyzer.visualize_readmission_rates(df)
        analyzer.build_predictive_model(df)
        analyzer.perform_clustering(df)
        
        analyzer.save_analysis_report()
        
        db_connector.close()
        
        logger.info("Analysis complete")
        
        return 0
    except Exception as e:
        import traceback
        logger.error(f"Error in analysis process: {str(e)}")
        logger.error("Detailed traceback:")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())