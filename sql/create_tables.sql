CREATE DATABASE IF NOT EXISTS healthcare_readmission;
USE healthcare_readmission;

-- Patient demographics table
CREATE TABLE IF NOT EXISTS patients (
    patient_id VARCHAR(20) PRIMARY KEY,
    age INT NOT NULL,
    gender VARCHAR(10) NOT NULL,
    race VARCHAR(50),
    marital_status VARCHAR(20),
    language VARCHAR(20),
    socioeconomic_status VARCHAR(20),
    insurance_type VARCHAR(30),
    zip_code VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Hospital stay information
CREATE TABLE IF NOT EXISTS hospital_stays (
    stay_id VARCHAR(20) PRIMARY KEY,
    patient_id VARCHAR(20) NOT NULL,
    admission_date DATE NOT NULL,
    discharge_date DATE,
    length_of_stay INT,
    admission_type VARCHAR(20),
    discharge_disposition VARCHAR(50),
    admission_source VARCHAR(50),
    is_readmission BOOLEAN DEFAULT FALSE,
    previous_stay_id VARCHAR(20),
    days_since_last_discharge INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (previous_stay_id) REFERENCES hospital_stays(stay_id)
);

-- Diagnoses table
CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id VARCHAR(20) PRIMARY KEY,
    stay_id VARCHAR(20) NOT NULL,
    icd_code VARCHAR(20) NOT NULL,
    icd_version INT,
    description VARCHAR(255),
    diagnosis_type VARCHAR(20),
    is_primary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (stay_id) REFERENCES hospital_stays(stay_id)
);

-- Procedures table
CREATE TABLE IF NOT EXISTS procedures (
    procedure_id VARCHAR(20) PRIMARY KEY,
    stay_id VARCHAR(20) NOT NULL,
    procedure_code VARCHAR(20) NOT NULL,
    procedure_date DATE,
    description VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (stay_id) REFERENCES hospital_stays(stay_id)
);

-- Medications table
CREATE TABLE IF NOT EXISTS medications (
    medication_id VARCHAR(20) PRIMARY KEY,
    stay_id VARCHAR(20) NOT NULL,
    medication_name VARCHAR(100) NOT NULL,
    ndc_code VARCHAR(20),
    dosage VARCHAR(50),
    frequency VARCHAR(50),
    start_date DATE,
    end_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (stay_id) REFERENCES hospital_stays(stay_id)
);

-- Lab results table
CREATE TABLE IF NOT EXISTS lab_results (
    result_id VARCHAR(20) PRIMARY KEY,
    stay_id VARCHAR(20) NOT NULL,
    test_name VARCHAR(100) NOT NULL,
    test_date DATETIME,
    result_value VARCHAR(50),
    unit VARCHAR(20),
    reference_range VARCHAR(50),
    abnormal_flag VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (stay_id) REFERENCES hospital_stays(stay_id)
);

-- Comorbidities table (pre-existing conditions)
CREATE TABLE IF NOT EXISTS comorbidities (
    comorbidity_id VARCHAR(20) PRIMARY KEY,
    patient_id VARCHAR(20) NOT NULL,
    condition_name VARCHAR(100) NOT NULL,
    icd_code VARCHAR(20),
    diagnosed_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Create view for readmission analysis
CREATE OR REPLACE VIEW readmission_analysis AS
SELECT 
    p.patient_id,
    p.age,
    p.gender,
    p.race,
    p.socioeconomic_status,
    p.insurance_type,
    hs.stay_id,
    hs.admission_date,
    hs.discharge_date,
    hs.length_of_stay,
    hs.is_readmission,
    hs.days_since_last_discharge,
    CASE WHEN hs.days_since_last_discharge <= 30 THEN 1 ELSE 0 END AS is_30day_readmission,
    d.icd_code AS primary_diagnosis,
    d.description AS primary_diagnosis_description,
    COUNT(DISTINCT c.comorbidity_id) AS comorbidity_count,
    COUNT(DISTINCT m.medication_id) AS medication_count,
    COUNT(DISTINCT pr.procedure_id) AS procedure_count
FROM 
    patients p
JOIN 
    hospital_stays hs ON p.patient_id = hs.patient_id
LEFT JOIN 
    diagnoses d ON hs.stay_id = d.stay_id AND d.is_primary = TRUE
LEFT JOIN 
    comorbidities c ON p.patient_id = c.patient_id
LEFT JOIN 
    medications m ON hs.stay_id = m.stay_id
LEFT JOIN 
    procedures pr ON hs.stay_id = pr.stay_id
GROUP BY 
    p.patient_id, hs.stay_id;

-- Create index for faster queries
CREATE INDEX idx_hospital_stays_patient_id ON hospital_stays(patient_id);
CREATE INDEX idx_hospital_stays_readmission ON hospital_stays(is_readmission);
CREATE INDEX idx_diagnoses_stay_id ON diagnoses(stay_id);
CREATE INDEX idx_diagnoses_icd_code ON diagnoses(icd_code);
CREATE INDEX idx_patients_demographics ON patients(age, gender, race);