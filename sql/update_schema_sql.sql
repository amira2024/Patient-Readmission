-- Update database schema to accommodate specific dataset columns
-- Run this after create_tables.sql

USE healthcare_readmission;

ALTER TABLE patients
ADD COLUMN patient_name VARCHAR(100) AFTER patient_id,
ADD COLUMN blood_type VARCHAR(5) AFTER gender;

ALTER TABLE hospital_stays
ADD COLUMN room_number VARCHAR(20) AFTER admission_source,
ADD COLUMN billing_amount DECIMAL(10, 2) AFTER room_number;

ALTER TABLE hospital_stays
ADD COLUMN n_outpatient INT AFTER length_of_stay,
ADD COLUMN n_inpatient INT AFTER n_outpatient,
ADD COLUMN n_emergency INT AFTER n_inpatient;

CREATE TABLE IF NOT EXISTS lab_test_results (
    test_id VARCHAR(20) PRIMARY KEY,
    patient_id VARCHAR(20) NOT NULL,
    stay_id VARCHAR(20) NOT NULL,
    test_name VARCHAR(50) NOT NULL,
    test_result VARCHAR(50) NOT NULL,
    test_date DATE,
    normal_range VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (stay_id) REFERENCES hospital_stays(stay_id)
);

CREATE INDEX idx_test_results_stay ON lab_test_results(stay_id);
CREATE INDEX idx_test_results_name ON lab_test_results(test_name);

CREATE TABLE IF NOT EXISTS facilities (
    facility_id VARCHAR(20) PRIMARY KEY,
    facility_name VARCHAR(255) NOT NULL,
    state VARCHAR(2) NOT NULL,
    measure_name VARCHAR(100),
    excess_readmission_ratio DECIMAL(5, 4),
    predicted_readmission_rate DECIMAL(5, 4),
    expected_readmission_rate DECIMAL(5, 4),
    number_of_readmissions INT,
    start_date DATE,
    end_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS drug_performance (
    drug_id VARCHAR(20) PRIMARY KEY,
    drug_name VARCHAR(100) NOT NULL,
    condition VARCHAR(100),
    indication VARCHAR(255),
    drug_type VARCHAR(20), -- generic or brand
    reviews_count INT,
    effectiveness_score DECIMAL(5, 2),
    ease_of_use_score DECIMAL(5, 2),
    satisfaction_score DECIMAL(5, 2),
    additional_info TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS medication_drug_performance (
    id VARCHAR(20) PRIMARY KEY,
    medication_id VARCHAR(20) NOT NULL,
    drug_id VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (medication_id) REFERENCES medications(medication_id),
    FOREIGN KEY (drug_id) REFERENCES drug_performance(drug_id)
);

CREATE OR REPLACE VIEW comprehensive_readmission_analysis AS
SELECT 
    p.patient_id,
    p.patient_name,
    p.age,
    p.gender,
    p.blood_type,
    p.race,
    p.socioeconomic_status,
    p.insurance_type,
    hs.stay_id,
    hs.admission_date,
    hs.discharge_date,
    hs.length_of_stay,
    hs.n_outpatient,
    hs.n_inpatient,
    hs.n_emergency,
    hs.admission_type,
    hs.room_number,
    hs.billing_amount,
    hs.is_readmission,
    hs.days_since_last_discharge,
    CASE WHEN hs.days_since_last_discharge <= 30 THEN 1 ELSE 0 END AS is_30day_readmission,
    d.icd_code AS primary_diagnosis,
    d.description AS primary_diagnosis_description,
    COUNT(DISTINCT c.comorbidity_id) AS comorbidity_count,
    COUNT(DISTINCT m.medication_id) AS medication_count,
    COUNT(DISTINCT pr.procedure_id) AS procedure_count,
    MAX(CASE WHEN ltr.test_name = 'Glucose' THEN ltr.test_result END) AS glucose_result,
    MAX(CASE WHEN ltr.test_name = 'A1C' THEN ltr.test_result END) AS a1c_result,
    AVG(dp.effectiveness_score) AS avg_medication_effectiveness,
    AVG(dp.satisfaction_score) AS avg_medication_satisfaction
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
LEFT JOIN
    lab_test_results ltr ON hs.stay_id = ltr.stay_id
LEFT JOIN
    medication_drug_performance mdp ON m.medication_id = mdp.medication_id
LEFT JOIN
    drug_performance dp ON mdp.drug_id = dp.drug_id
GROUP BY 
    p.patient_id, hs.stay_id, d.icd_code, d.description;
