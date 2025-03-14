"""
load.py - Data loading script for Patient Readmission Analysis project

This script loads the transformed data into a MySQL database according to the
schema defined in create_tables.sql.
"""

import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import argparse
import re
import uuid
import mysql.connector
from mysql.connector import Error
from typing import Dict, List, Optional, Union
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/load.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

os.makedirs('logs', exist_ok=True)


class DatabaseLoader:
    """Class to handle loading data into a MySQL database"""
    
    def __init__(self, config_file: str = 'config/database.yaml'):
        """Initialize the loader with database configuration"""
        self.config_file = config_file
        self.connection = None
        self.cursor = None
        
        self.config = self._load_config()
        
        self._connect()
        
        self.load_operations = []
    
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
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    return config
            else:
                logger.warning(f"Config file {self.config_file} not found, creating default")
                with open(self.config_file, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
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
                
                self.cursor.execute("SHOW DATABASES LIKE %s", (self.config.get('database'),))
                if not self.cursor.fetchone():
                    logger.info(f"Creating database {self.config.get('database')}")
                    self.cursor.execute(f"CREATE DATABASE {self.config.get('database')}")
                    self.connection.database = self.config.get('database')
                
                self._create_tables_if_not_exist()
        except Error as e:
            logger.error(f"Error connecting to MySQL: {str(e)}")
            raise
    
    def _create_tables_if_not_exist(self) -> None:
        """Create tables if they don't exist using create_tables.sql"""
        try:
            sql_file_path = 'sql/create_tables.sql'
            if not os.path.exists(sql_file_path):
                logger.error(f"SQL file {sql_file_path} not found")
                return
            
            with open(sql_file_path, 'r') as f:
                sql_script = f.read()
            
            statements = sql_script.split(';')
            
            for statement in statements:
                if statement.strip():
                    try:
                        self.cursor.execute(statement)
                        logger.debug(f"Executed SQL: {statement[:50]}...")
                    except Error as e:
                        logger.warning(f"Error executing SQL: {str(e)}")
            
            self.connection.commit()
            logger.info("Tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
    
    def get_latest_processed_file(self, name_pattern: str) -> Optional[str]:
        """Get the path to the latest processed file matching the pattern"""
        processed_dir = 'data/processed'
        pattern = re.compile(f"^{name_pattern}_\\d{{8}}_\\d{{6}}\\.csv$")
        files = [f for f in os.listdir(processed_dir) if pattern.match(f)]
        
        if not files:
            logger.warning(f"No processed files found matching pattern: {name_pattern}")
            return None
        
        latest_file = sorted(files)[-1]
        return os.path.join(processed_dir, latest_file)
    
    def load_patients_data(self, file_path: str) -> int:
        """Load patient demographic data into the patients table"""
        try:
            logger.info(f"Loading patient data from {file_path}")
            
            df = pd.read_csv(file_path)
            
            column_mapping = {
                'patient_id': 'patient_id',
                'age': 'age',
                'gender': 'gender',
                'race': 'race',
                'marital_status': 'marital_status',
                'language': 'language',
                'socioeconomic_status': 'socioeconomic_status',
                'insurance': 'insurance_type',
                'zipcode': 'zip_code'
            }
            
            columns_to_select = [col for col in column_mapping.keys() if col in df.columns]
            if not columns_to_select:
                logger.error("No matching columns found for patients table")
                return 0
            
            patients_df = df[columns_to_select].copy()
            patients_df.rename(columns={k: v for k, v in column_mapping.items() if k in columns_to_select}, inplace=True)
            
            required_columns = ['patient_id', 'age', 'gender']
            for col in required_columns:
                if col not in patients_df.columns:
                    logger.error(f"Required column {col} not found in data")
                    return 0
            
            if 'patient_id' not in patients_df.columns:
                patients_df['patient_id'] = [str(uuid.uuid4()) for _ in range(len(patients_df))]
            
            rows_inserted = 0
            for _, row in patients_df.iterrows():
                # Check if patient already exists
                self.cursor.execute(
                    "SELECT patient_id FROM patients WHERE patient_id = %s",
                    (row['patient_id'],)
                )
                if self.cursor.fetchone():
                    update_cols = [f"{col} = %s" for col in patients_df.columns if col != 'patient_id']
                    query = f"UPDATE patients SET {', '.join(update_cols)} WHERE patient_id = %s"
                    values = [row[col] for col in patients_df.columns if col != 'patient_id'] + [row['patient_id']]
                    self.cursor.execute(query, values)
                else:
                    # Insert new patient
                    placeholders = ', '.join(['%s'] * len(patients_df.columns))
                    query = f"INSERT INTO patients ({', '.join(patients_df.columns)}) VALUES ({placeholders})"
                    values = [row[col] for col in patients_df.columns]
                    self.cursor.execute(query, values)
                    rows_inserted += 1
            
            self.connection.commit()
            logger.info(f"Inserted {rows_inserted} patients")
            
            self.load_operations.append({
                'table': 'patients',
                'file': file_path,
                'rows_processed': len(patients_df),
                'rows_inserted': rows_inserted,
                'timestamp': datetime.now().isoformat()
            })
            
            return rows_inserted
        except Exception as e:
            logger.error(f"Error loading patient data: {str(e)}")
            self.connection.rollback()
            return 0
    
    def load_hospital_stays_data(self, file_path: str) -> int:
        """Load hospital stay data into the hospital_stays table"""
        try:
            logger.info(f"Loading hospital stays data from {file_path}")
            
            df = pd.read_csv(file_path)
            
            # Map columns to match database schema
            column_mapping = {
                'stay_id': 'stay_id',
                'patient_id': 'patient_id',
                'admission_date': 'admission_date',
                'discharge_date': 'discharge_date',
                'length_of_stay': 'length_of_stay',
                'admission_type': 'admission_type',
                'discharge_disposition': 'discharge_disposition',
                'admission_source': 'admission_source',
                'is_readmission': 'is_readmission',
                'previous_stay_id': 'previous_stay_id',
                'days_since_last_discharge': 'days_since_last_discharge'
            }
            
            columns_to_select = [col for col in column_mapping.keys() if col in df.columns]
            if not columns_to_select:
                logger.error("No matching columns found for hospital_stays table")
                return 0
            
            stays_df = df[columns_to_select].copy()
            stays_df.rename(columns={k: v for k, v in column_mapping.items() if k in columns_to_select}, inplace=True)
            
            required_columns = ['patient_id', 'admission_date']
            for col in required_columns:
                if col not in stays_df.columns:
                    logger.error(f"Required column {col} not found in data")
                    return 0
            
            if 'stay_id' not in stays_df.columns:
                stays_df['stay_id'] = [str(uuid.uuid4()) for _ in range(len(stays_df))]
            
            date_cols = ['admission_date', 'discharge_date']
            for col in date_cols:
                if col in stays_df.columns:
                    stays_df[col] = pd.to_datetime(stays_df[col]).dt.strftime('%Y-%m-%d')
            
            rows_inserted = 0
            for _, row in stays_df.iterrows():
                try:
                    self.cursor.execute(
                        "SELECT stay_id FROM hospital_stays WHERE stay_id = %s",
                        (row['stay_id'],)
                    )
                    if self.cursor.fetchone():
                        update_cols = [f"{col} = %s" for col in stays_df.columns if col != 'stay_id']
                        query = f"UPDATE hospital_stays SET {', '.join(update_cols)} WHERE stay_id = %s"
                        values = [row[col] for col in stays_df.columns if col != 'stay_id'] + [row['stay_id']]
                        self.cursor.execute(query, values)
                    else:
                        placeholders = ', '.join(['%s'] * len(stays_df.columns))
                        query = f"INSERT INTO hospital_stays ({', '.join(stays_df.columns)}) VALUES ({placeholders})"
                        values = [row[col] for col in stays_df.columns]
                        self.cursor.execute(query, values)
                        rows_inserted += 1
                except Error as e:
                    logger.warning(f"Error inserting hospital stay: {str(e)}")
                    continue
            
            self.connection.commit()
            logger.info(f"Inserted {rows_inserted} hospital stays")
            
            self.load_operations.append({
                'table': 'hospital_stays',
                'file': file_path,
                'rows_processed': len(stays_df),
                'rows_inserted': rows_inserted,
                'timestamp': datetime.now().isoformat()
            })
            
            return rows_inserted
        except Exception as e:
            logger.error(f"Error loading hospital stays data: {str(e)}")
            self.connection.rollback()
            return 0
    
    def load_diagnoses_data(self, file_path: str) -> int:
        """Load diagnoses data into the diagnoses table"""
        try:
            logger.info(f"Loading diagnoses data from {file_path}")
            
            df = pd.read_csv(file_path)
            
            column_mapping = {
                'diagnosis_id': 'diagnosis_id',
                'stay_id': 'stay_id',
                'icd_code': 'icd_code',
                'icd_version': 'icd_version',
                'description': 'description',
                'diagnosis_type': 'diagnosis_type',
                'is_primary': 'is_primary'
            }
            
            columns_to_select = [col for col in column_mapping.keys() if col in df.columns]
            if not columns_to_select:
                logger.error("No matching columns found for diagnoses table")
                return 0
            
            diagnoses_df = df[columns_to_select].copy()
            diagnoses_df.rename(columns={k: v for k, v in column_mapping.items() if k in columns_to_select}, inplace=True)
            
            required_columns = ['stay_id', 'icd_code']
            for col in required_columns:
                if col not in diagnoses_df.columns:
                    logger.error(f"Required column {col} not found in data")
                    return 0
            
            if 'diagnosis_id' not in diagnoses_df.columns:
                diagnoses_df['diagnosis_id'] = [str(uuid.uuid4()) for _ in range(len(diagnoses_df))]
            
            if 'is_primary' not in diagnoses_df.columns:
                diagnoses_df['is_primary'] = False
            
            rows_inserted = 0
            for _, row in diagnoses_df.iterrows():
                try:
                    self.cursor.execute(
                        "SELECT diagnosis_id FROM diagnoses WHERE diagnosis_id = %s",
                        (row['diagnosis_id'],)
                    )
                    if self.cursor.fetchone():
                        update_cols = [f"{col} = %s" for col in diagnoses_df.columns if col != 'diagnosis_id']
                        query = f"UPDATE diagnoses SET {', '.join(update_cols)} WHERE diagnosis_id = %s"
                        values = [row[col] for col in diagnoses_df.columns if col != 'diagnosis_id'] + [row['diagnosis_id']]
                        self.cursor.execute(query, values)
                    else:
                        placeholders = ', '.join(['%s'] * len(diagnoses_df.columns))
                        query = f"INSERT INTO diagnoses ({', '.join(diagnoses_df.columns)}) VALUES ({placeholders})"
                        values = [row[col] for col in diagnoses_df.columns]
                        self.cursor.execute(query, values)
                        rows_inserted += 1
                except Error as e:
                    logger.warning(f"Error inserting diagnosis: {str(e)}")
                    continue
            
            self.connection.commit()
            logger.info(f"Inserted {rows_inserted} diagnoses")
            
            self.load_operations.append({
                'table': 'diagnoses',
                'file': file_path,
                'rows_processed': len(diagnoses_df),
                'rows_inserted': rows_inserted,
                'timestamp': datetime.now().isoformat()
            })
            
            return rows_inserted
        except Exception as e:
            logger.error(f"Error loading diagnoses data: {str(e)}")
            self.connection.rollback()
            return 0
    
    def save_load_report(self) -> None:
        """Save load operations report to JSON file"""
        report_dir = 'data/reports'
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, f"load_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'operations': self.load_operations,
            'total_tables': len(set(op['table'] for op in self.load_operations)),
            'total_rows_inserted': sum(op['rows_inserted'] for op in self.load_operations)
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved load report to {report_path}")
    
    def close(self) -> None:
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            if self.cursor:
                self.cursor.close()
            self.connection.close()
            logger.info("Database connection closed")


def main():
    """Main function to run the data loading process"""
    parser = argparse.ArgumentParser(description='Load processed data into MySQL database')
    parser.add_argument('--config', default='config/database.yaml', help='Path to database config file')
    args = parser.parse_args()
    
    try:
        loader = DatabaseLoader(config_file=args.config)
        
        merged_file = loader.get_latest_processed_file('merged_dataset')
        if merged_file:
            loader.load_patients_data(merged_file)
            loader.load_hospital_stays_data(merged_file)
            loader.load_diagnoses_data(merged_file)
        else:
            logger.error("No processed data files found")
        
        loader.save_load_report()
        
        loader.close()
        
        return 0
    except Exception as e:
        logger.error(f"Error in data loading process: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())