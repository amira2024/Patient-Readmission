"""
transform.py - Data transformation script for Patient Readmission Analysis project

This script cleans, preprocesses, and transforms the raw datasets into a
structured format suitable for analysis and database loading.
"""

import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import argparse
from typing import Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/transform.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.makedirs('logs', exist_ok=True)


class DataTransformer:
    """Class to handle data transformation and preprocessing"""
    
    def __init__(self, data_dir: str = 'data'):
        """Initialize the transformer with data directory path"""
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, 'raw')
        self.processed_dir = os.path.join(data_dir, 'processed')
        
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.transformations: Dict[str, Dict] = {}
        
        self.encoders = {}
        self.scalers = {}
    
    def get_latest_raw_file(self, source_name: str) -> Optional[str]:
        """Get the path to the latest raw file for a given source"""
        pattern = re.compile(f"^{source_name}_\\d{{8}}_\\d{{6}}\\.csv$")
        files = [f for f in os.listdir(self.raw_dir) if pattern.match(f)]
        
        if not files:
            logger.warning(f"No raw files found for source: {source_name}")
            return None
        
        latest_file = sorted(files)[-1]
        return os.path.join(self.raw_dir, latest_file)
    
    def load_raw_data(self, source_name: str) -> Optional[pd.DataFrame]:
        """Load the latest raw data file for a given source"""
        file_path = self.get_latest_raw_file(source_name)
        if file_path is None:
            return None
        
        logger.info(f"Loading raw data from {file_path}")
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Failed to load raw data from {file_path}: {str(e)}")
            return None
    
    def clean_healthcare_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the main healthcare dataset"""
        logger.info("Cleaning healthcare dataset")
        
        df_clean = df.copy()
        
        transform_steps = []
        
        missing_before = df_clean.isnull().sum().sum()
        
        # For numerical columns, fill with median
        num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        missing_after = df_clean.isnull().sum().sum()
        transform_steps.append(f"Filled {missing_before - missing_after} missing values")
        
        date_cols = [col for col in df_clean.columns if 'date' in col.lower()]
        for col in date_cols:
            if col in df_clean.columns:
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col])
                    transform_steps.append(f"Converted {col} to datetime")
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to datetime: {str(e)}")
        
        # Add readmission indicator and days since last admission
        if all(col in df_clean.columns for col in ['patient_id', 'admission_date']):
            # Sort by patient_id and admission_date
            df_clean = df_clean.sort_values(['patient_id', 'admission_date'])
            
            # Add a column indicating if this is a readmission
            df_clean['is_readmission'] = df_clean.duplicated(subset=['patient_id'], keep='first')
            
            # Calculate days since last discharge for the same patient
            df_clean['previous_admission_date'] = df_clean.groupby('patient_id')['admission_date'].shift(1)
            df_clean['previous_discharge_date'] = df_clean.groupby('patient_id')['discharge_date'].shift(1)
            
            # Calculate days since last discharge
            mask = ~df_clean['previous_discharge_date'].isna()
            df_clean.loc[mask, 'days_since_last_discharge'] = (
                df_clean.loc[mask, 'admission_date'] - 
                df_clean.loc[mask, 'previous_discharge_date']
            ).dt.days
            
            transform_steps.append("Added readmission indicators and days between admissions")
        
        self.transformations['healthcare_dataset'] = {
            'rows_before': len(df),
            'rows_after': len(df_clean),
            'steps': transform_steps,
            'timestamp': datetime.now().isoformat()
        }
        
        return df_clean
    
    def clean_readmissions_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the hospital readmissions dataset"""
        logger.info("Cleaning readmissions dataset")
        
        df_clean = df.copy()
        
        transform_steps = []
        
        missing_before = df_clean.isnull().sum().sum()
        
        num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        missing_after = df_clean.isnull().sum().sum()
        transform_steps.append(f"Filled {missing_before - missing_after} missing values")
        
        for col in cat_cols:
            if df_clean[col].nunique() < 10:  # For low-cardinality categories
                le = LabelEncoder()
                df_clean[f"{col}_encoded"] = le.fit_transform(df_clean[col])
                self.encoders[col] = le
                transform_steps.append(f"Label encoded {col}")
        
        self.transformations['readmissions_dataset'] = {
            'rows_before': len(df),
            'rows_after': len(df_clean),
            'steps': transform_steps,
            'timestamp': datetime.now().isoformat()
        }
        
        return df_clean
    
    def clean_drug_performance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the drug performance dataset"""
        logger.info("Cleaning drug performance dataset")
        
        df_clean = df.copy()
        
        transform_steps = []
        
        missing_before = df_clean.isnull().sum().sum()
        
        num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        missing_after = df_clean.isnull().sum().sum()
        transform_steps.append(f"Filled {missing_before - missing_after} missing values")
        
        scaler = StandardScaler()
        num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        if len(num_cols) > 0:
            df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
            self.scalers['drug_performance'] = scaler
            transform_steps.append(f"Standardized {len(num_cols)} numerical columns")
        
        self.transformations['drug_performance_dataset'] = {
            'rows_before': len(df),
            'rows_after': len(df_clean),
            'steps': transform_steps,
            'timestamp': datetime.now().isoformat()
        }
        
        return df_clean
    
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple datasets based on common keys"""
        logger.info(f"Merging {len(datasets)} datasets")
        
        if 'healthcare_dataset' in datasets and 'readmissions_dataset' in datasets:
            common_cols = set(datasets['healthcare_dataset'].columns) & set(datasets['readmissions_dataset'].columns)
            
            if 'patient_id' in common_cols:
                logger.info("Merging datasets on patient_id")
                merged_df = pd.merge(
                    datasets['healthcare_dataset'],
                    datasets['readmissions_dataset'],
                    on='patient_id',
                    how='outer',
                    suffixes=('', '_readmission')
                )
            else:
                logger.warning("No common patient identifier found, cannot merge datasets")
                merged_df = datasets['healthcare_dataset']
        else:
            key = list(datasets.keys())[0]
            logger.info(f"Only {key} dataset available, using it as the base")
            merged_df = datasets[key]
        
        return merged_df
    
    def create_features_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for machine learning models"""
        logger.info("Creating features for machine learning")
        
        df_features = df.copy()
        
        transform_steps = []
        
        if 'days_since_last_discharge' in df_features.columns:
            df_features['readmitted_30_days'] = (df_features['days_since_last_discharge'] <= 30) & (df_features['days_since_last_discharge'] >= 0)
            transform_steps.append("Created 30-day readmission target variable")
        
        # One-hot encode categorical variables with many categories
        cat_cols = df_features.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df_features[col].nunique() > 10:  # For high-cardinality categories
                # Create one-hot encoder
                ohe = OneHotEncoder(sparse=False, drop='first')
                encoded = ohe.fit_transform(df_features[[col]])
                
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=[f"{col}_{cat}" for cat in ohe.categories_[0][1:]],
                    index=df_features.index
                )
                
                # Join the encoded features
                df_features = pd.concat([df_features, encoded_df], axis=1)
                
                # Store the encoder
                self.encoders[f"{col}_onehot"] = ohe
                transform_steps.append(f"One-hot encoded {col}")
        
        if 'age' in df_features.columns and 'length_of_stay' in df_features.columns:
            df_features['age_los_interaction'] = df_features['age'] * df_features['length_of_stay']
            transform_steps.append("Created age x length_of_stay interaction feature")
        
        self.transformations['ml_features'] = {
            'steps': transform_steps,
            'timestamp': datetime.now().isoformat()
        }
        
        return df_features
    
    def save_processed_data(self, df: pd.DataFrame, name: str) -> str:
        """Save processed DataFrame to CSV"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(self.processed_dir, f"{name}_{timestamp}.csv")
        
        logger.info(f"Saving processed data to {file_path}")
        df.to_csv(file_path, index=False)
        
        return file_path
    
    def save_transformation_metadata(self) -> None:
        """Save transformation metadata to JSON file"""
        metadata_path = os.path.join(self.processed_dir, 'transformation_metadata.json')
        
        logger.info(f"Saving transformation metadata to {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(self.transformations, f, indent=2)
        
        # Also save encoders and scalers for later use
        if self.encoders:
            from joblib import dump
            encoders_path = os.path.join(self.processed_dir, 'encoders.joblib')
            dump(self.encoders, encoders_path)
            logger.info(f"Saved encoders to {encoders_path}")
        
        if self.scalers:
            from joblib import dump
            scalers_path = os.path.join(self.processed_dir, 'scalers.joblib')
            dump(self.scalers, scalers_path)
            logger.info(f"Saved scalers to {scalers_path}")


def main():
    """Main function to run the transformation process"""
    parser = argparse.ArgumentParser(description='Transform healthcare data')
    parser.add_argument('--data-dir', default='data', help='Directory containing raw data')
    args = parser.parse_args()
    
    transformer = DataTransformer(data_dir=args.data_dir)
    
    datasets = {}
    for source in ['healthcare_dataset', 'hospital_readmissions', 'drug_performance']:
        df = transformer.load_raw_data(source)
        if df is not None:
            datasets[source] = df
    
    if not datasets:
        logger.error("No datasets loaded, aborting")
        return 1
    
    cleaned_datasets = {}
    if 'healthcare_dataset' in datasets:
        cleaned_datasets['healthcare_dataset'] = transformer.clean_healthcare_dataset(datasets['healthcare_dataset'])
    
    if 'hospital_readmissions' in datasets:
        cleaned_datasets['readmissions_dataset'] = transformer.clean_readmissions_dataset(datasets['hospital_readmissions'])
    
    if 'drug_performance' in datasets:
        cleaned_datasets['drug_performance_dataset'] = transformer.clean_drug_performance_dataset(datasets['drug_performance'])
    
    merged_df = transformer.merge_datasets(cleaned_datasets)
    
    ml_df = transformer.create_features_for_ml(merged_df)
    
    transformer.save_processed_data(merged_df, 'merged_dataset')
    transformer.save_processed_data(ml_df, 'ml_features_dataset')
    
    transformer.save_transformation_metadata()
    
    logger.info("Transformation complete")
    return 0


if __name__ == "__main__":
    exit(main())