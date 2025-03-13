CREATE TABLE patients (
    patient_id VARCHAR(50) PRIMARY KEY,
    age INT,
    gender VARCHAR(10),
    race VARCHAR(50),
    insurance VARCHAR(50),
    zipcode VARCHAR(10),
    county VARCHAR(50)
);

CREATE TABLE admissions (
    admission_id VARCHAR(50) PRIMARY KEY,
    patient_id VARCHAR(50),
    admission_date DATE,
    discharge_date DATE,
    length_of_stay INT,
    admission_type VARCHAR(50),
    department VARCHAR(50),
    primary_diagnosis VARCHAR(100),
    secondary_diagnosis VARCHAR(100),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE TABLE treatments (
    treatment_id VARCHAR(50) PRIMARY KEY,
    admission_id VARCHAR(50),
    treatment_type VARCHAR(100),
    medication VARCHAR(100),
    dosage VARCHAR(50),
    start_date DATE,
    end_date DATE,
    FOREIGN KEY (admission_id) REFERENCES admissions(admission_id)
);


CREATE TABLE readmissions (
    readmission_id VARCHAR(50) PRIMARY KEY,
    patient_id VARCHAR(50),
    initial_admission_id VARCHAR(50),
    readmission_date DATE,
    days_since_discharge INT,
    reason VARCHAR(100),
    preventable BOOLEAN,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (initial_admission_id) REFERENCES admissions(admission_id)
);