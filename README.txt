Healthcare: Patient Readmission Analysis
Project Overview
This project analyzes hospital readmission data to identify key factors contributing to frequent patient readmissions. By leveraging data science techniques, machine learning models, and interactive visualizations, the project aims to help healthcare facilities reduce readmission rates and improve patient outcomes.
Show Image
Business Problem
Hospital readmissions represent a significant challenge in healthcare:

High readmission rates are associated with increased healthcare costs
The Centers for Medicare & Medicaid Services (CMS) penalizes hospitals with excessive readmission rates
Readmissions often indicate suboptimal patient care or discharge planning
Identifying high-risk patients can help target interventions effectively

This project develops a comprehensive analysis system to identify patterns in readmission data and build predictive models to forecast which patients are at highest risk for readmission.
Data Sources
The project utilizes several healthcare datasets:

Patient Demographics and Hospital Stays Dataset

Patient demographic information (age, gender, race, etc.)
Hospital stay details (admission dates, length of stay, etc.)
Diagnosis and procedure information


Diabetes Readmission Dataset

Specific diabetes patient data with readmission outcomes
Treatment details and lab test results
Previous visit history


Hospital Performance Metrics Dataset

Facility-level readmission metrics
Benchmark data for comparison


Drug Performance Dataset

Medication effectiveness ratings
Patient satisfaction with medications
Drug type information



Tech Stack
This project leverages a comprehensive technology stack:

Database: MySQL for structured data storage and querying
ETL Pipeline: Python scripts for data extraction, transformation, and loading
Data Analysis: Pandas, NumPy for data manipulation and analysis
Machine Learning: Scikit-learn, XGBoost for predictive modeling
Visualization: Matplotlib, Seaborn for static visualizations, Tableau for interactive dashboards
Version Control: Git for code management
Documentation: Markdown for project documentation

Project Structure
Copypatient_readmission_analysis/
│
├── data/                        # Data directory
│   ├── raw/                     # Raw data files
│   ├── processed/               # Processed datasets
│   └── reports/                 # Data quality reports
│
├── scripts/                     # Python scripts
│   ├── extract.py               # Data extraction from various sources
│   ├── transform.py             # Data cleaning and transformation
│   ├── load.py                  # Database loading
│   ├── analysis.py              # Data analysis and visualization
│   └── ml_model.py              # Machine learning modeling
│
├── sql/                         # SQL scripts
│   ├── create_tables.sql        # Database schema definition
│   └── insert_data.sql          # Sample data insertion
│
├── notebooks/                   # Jupyter notebooks
│   └── exploratory_data_analysis.ipynb  # EDA notebook
│
├── tableau/                     # Tableau dashboards
│   └── readmission_dashboard.twbx  # Main dashboard file
│
├── models/                      # Saved machine learning models
│   ├── random_forest_model.joblib
│   └── preprocessor.joblib
│
├── output/                      # Analysis outputs
│   ├── figures/                 # Generated visualizations
│   └── model_evaluation/        # Model performance metrics
│
├── docs/                        # Documentation
│   └── images/                  # Documentation images
│
├── logs/                        # Application logs
│
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
Key Features
1. ETL Pipeline

Data extraction from multiple sources (CSV files, APIs)
Comprehensive data cleaning and preprocessing
Structured database loading with data integrity checks

2. Exploratory Data Analysis

In-depth analysis of readmission patterns
Demographic and clinical factor exploration
Time-based readmission analysis

3. Machine Learning Models

Readmission risk prediction models
Feature importance analysis
Model evaluation and comparison
Hyperparameter optimization

4. Interactive Dashboards

Tableau visualizations for key metrics
Filterable by various dimensions
Drill-down capabilities for detailed analysis

Machine Learning Insights
The project develops several machine learning models to predict patient readmission risk:
ModelROC AUCPrecisionRecallF1 ScoreLogistic Regression0.780.710.650.68Random Forest0.830.740.720.73Gradient Boosting0.850.760.750.75XGBoost0.860.770.760.76
Key predictive factors for readmission include:

Length of hospital stay
Number of previous hospitalizations
Specific diagnoses (CHF, COPD, Diabetes)
Patient age
Number of medications
Specific medication types

Visualizations
The project includes a variety of visualizations to understand readmission patterns:

Readmission Rate by Age Group

Shows higher rates among elderly patients


Readmission by Length of Stay

Both very short and very long stays correlate with higher readmission


Time Between Discharge and Readmission

Histogram showing when readmissions typically occur


Top Diagnoses by Readmission Rate

Identifying conditions with highest readmission risk


Readmission Trends Over Time

Seasonal and temporal patterns in readmission rates



Installation and Setup
Prerequisites

Python 3.8+
MySQL 8.0+
Tableau Desktop (for dashboard modifications)

Setup Instructions

Clone the repository:

bashCopygit clone https://github.com/amira2024/patient_readmission_analysis.git
cd patient_readmission_analysis

Create a virtual environment and install dependencies:

bashCopypython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Set up the MySQL database:

bashCopymysql -u root -p < sql/create_tables.sql

Create a database configuration file at config/database.yaml:

yamlCopyhost: localhost
database: healthcare_readmission
user: your_username
password: your_password
port: 3306

Run the ETL pipeline:

bashCopypython scripts/extract.py
python scripts/transform.py
python scripts/load.py

Run analysis and modeling:

bashCopypython scripts/analysis.py
python scripts/ml_model.py
Usage Examples
Exploratory Data Analysis
bashCopyjupyter notebook notebooks/exploratory_data_analysis.ipynb
Running the ML Pipeline
bashCopypython scripts/ml_model.py --data-path data/processed/ml_features_dataset_latest.csv
Connecting to Tableau Dashboard

Open Tableau Desktop
Connect to data source using MySQL connection
Open tableau/readmission_dashboard.twbx

Future Enhancements
Potential future improvements to the project:

Real-time Processing

Implement streaming data pipeline for real-time readmission risk assessment


Advanced ML Models

Incorporate deep learning models for improved prediction accuracy
Add time-series forecasting for readmission trend prediction


Integration Capabilities

Develop API endpoints for integration with hospital EHR systems
Create alerting system for high-risk patients


Additional Data Sources

Incorporate social determinants of health data
Add medication adherence information



Conclusion
This healthcare readmission analysis project demonstrates a comprehensive approach to understanding and predicting patient readmissions. By leveraging a robust data pipeline, advanced analytics, and machine learning techniques, the project provides actionable insights that can help healthcare providers reduce readmission rates and improve patient outcomes.
The combination of SQL, Python, machine learning, and Tableau visualizations showcases proficiency in essential data engineering and data science skills, making this project a valuable addition to a professional portfolio.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact
For questions or inquiries about this project, please contact amiragarba13@gmail.com.