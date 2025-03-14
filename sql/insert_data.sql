-- Healthcare: Patient Readmission Analysis
-- insert_data.sql
-- Sample data for testing and development

USE healthcare_readmission;

INSERT INTO patients 
(patient_id, age, gender, race, marital_status, language, socioeconomic_status, insurance_type, zip_code)
VALUES
('P10001', 67, 'Male', 'Caucasian', 'Married', 'English', 'Middle', 'Medicare', '10001'),
('P10002', 45, 'Female', 'African American', 'Single', 'English', 'Low', 'Medicaid', '20002'),
('P10003', 72, 'Male', 'Caucasian', 'Widowed', 'English', 'Middle', 'Medicare', '30003'),
('P10004', 34, 'Female', 'Hispanic', 'Married', 'Spanish', 'Low', 'Blue Cross', '40004'),
('P10005', 56, 'Male', 'Asian', 'Married', 'Mandarin', 'High', 'UnitedHealthcare', '50005'),
('P10006', 61, 'Female', 'Caucasian', 'Divorced', 'English', 'Middle', 'Aetna', '60006'),
('P10007', 28, 'Male', 'African American', 'Single', 'English', 'Low', 'Cigna', '70007'),
('P10008', 83, 'Female', 'Caucasian', 'Widowed', 'English', 'Middle', 'Medicare', '80008'),
('P10009', 42, 'Male', 'Hispanic', 'Married', 'Spanish', 'Low', 'Medicaid', '90009'),
('P10010', 75, 'Female', 'Caucasian', 'Married', 'English', 'Middle', 'Medicare', '10010');

INSERT INTO hospital_stays 
(stay_id, patient_id, admission_date, discharge_date, length_of_stay, admission_type, discharge_disposition, admission_source, is_readmission, days_since_last_discharge)
VALUES
('S20001', 'P10001', '2023-01-05', '2023-01-12', 7, 'Emergency', 'Home', 'Emergency Room', FALSE, NULL),
('S20002', 'P10002', '2023-01-10', '2023-01-14', 4, 'Elective', 'Home', 'Physician Referral', FALSE, NULL),
('S20003', 'P10003', '2023-01-15', '2023-01-25', 10, 'Emergency', 'Skilled Nursing Facility', 'Transfer', FALSE, NULL),
('S20004', 'P10004', '2023-01-20', '2023-01-24', 4, 'Urgent', 'Home', 'Physician Referral', FALSE, NULL),
('S20005', 'P10005', '2023-01-25', '2023-02-02', 8, 'Elective', 'Home', 'Outpatient', FALSE, NULL),
('S20006', 'P10001', '2023-02-12', '2023-02-18', 6, 'Emergency', 'Home', 'Emergency Room', TRUE, 31),
('S20007', 'P10006', '2023-02-15', '2023-02-20', 5, 'Elective', 'Home', 'Physician Referral', FALSE, NULL),
('S20008', 'P10003', '2023-02-20', '2023-02-28', 8, 'Emergency', 'Rehabilitation', 'Emergency Room', TRUE, 26),
('S20009', 'P10007', '2023-02-25', '2023-03-01', 4, 'Urgent', 'Home', 'Outpatient', FALSE, NULL),
('S20010', 'P10008', '2023-03-01', '2023-03-12', 11, 'Emergency', 'Skilled Nursing Facility', 'Emergency Room', FALSE, NULL),
('S20011', 'P10002', '2023-03-05', '2023-03-08', 3, 'Urgent', 'Home', 'Physician Referral', TRUE, 50),
('S20012', 'P10009', '2023-03-10', '2023-03-15', 5, 'Elective', 'Home', 'Physician Referral', FALSE, NULL),
('S20013', 'P10005', '2023-03-15', '2023-03-20', 5, 'Emergency', 'Home', 'Emergency Room', TRUE, 41),
('S20014', 'P10010', '2023-03-20', '2023-03-28', 8, 'Emergency', 'Skilled Nursing Facility', 'Emergency Room', FALSE, NULL),
('S20015', 'P10004', '2023-03-25', '2023-03-28', 3, 'Urgent', 'Home', 'Physician Referral', TRUE, 60),
('S20016', 'P10006', '2023-04-02', '2023-04-08', 6, 'Emergency', 'Home', 'Emergency Room', TRUE, 41),
('S20017', 'P10008', '2023-04-05', '2023-04-15', 10, 'Emergency', 'Rehabilitation', 'Emergency Room', TRUE, 24),
('S20018', 'P10003', '2023-04-10', '2023-04-18', 8, 'Urgent', 'Home', 'Physician Referral', TRUE, 41),
('S20019', 'P10007', '2023-04-15', '2023-04-18', 3, 'Elective', 'Home', 'Outpatient', TRUE, 45),
('S20020', 'P10001', '2023-04-20', '2023-04-30', 10, 'Emergency', 'Skilled Nursing Facility', 'Emergency Room', TRUE, 61);

INSERT INTO diagnoses 
(diagnosis_id, stay_id, icd_code, icd_version, description, diagnosis_type, is_primary)
VALUES
('D30001', 'S20001', 'I50.9', 10, 'Heart failure, unspecified', 'Principal', TRUE),
('D30002', 'S20001', 'I10', 10, 'Essential (primary) hypertension', 'Secondary', FALSE),
('D30003', 'S20001', 'E11.9', 10, 'Type 2 diabetes mellitus without complications', 'Secondary', FALSE),
('D30004', 'S20002', 'J44.9', 10, 'Chronic obstructive pulmonary disease, unspecified', 'Principal', TRUE),
('D30005', 'S20002', 'J45.901', 10, 'Unspecified asthma with (acute) exacerbation', 'Secondary', FALSE),
('D30006', 'S20003', 'I63.9', 10, 'Cerebral infarction, unspecified', 'Principal', TRUE),
('D30007', 'S20003', 'I10', 10, 'Essential (primary) hypertension', 'Secondary', FALSE),
('D30008', 'S20004', 'K80.10', 10, 'Calculus of gallbladder with chronic cholecystitis without obstruction', 'Principal', TRUE),
('D30009', 'S20005', 'M17.9', 10, 'Osteoarthritis of knee, unspecified', 'Principal', TRUE),
('D30010', 'S20006', 'I50.9', 10, 'Heart failure, unspecified', 'Principal', TRUE),
('D30011', 'S20006', 'N17.9', 10, 'Acute kidney failure, unspecified', 'Secondary', FALSE),
('D30012', 'S20007', 'C50.911', 10, 'Malignant neoplasm of unspecified site of right female breast', 'Principal', TRUE),
('D30013', 'S20008', 'I63.9', 10, 'Cerebral infarction, unspecified', 'Principal', TRUE),
('D30014', 'S20009', 'K35.80', 10, 'Unspecified acute appendicitis', 'Principal', TRUE),
('D30015', 'S20010', 'S72.001A', 10, 'Fracture of unspecified part of neck of right femur, initial encounter', 'Principal', TRUE),
('D30016', 'S20011', 'J44.1', 10, 'Chronic obstructive pulmonary disease with (acute) exacerbation', 'Principal', TRUE),
('D30017', 'S20012', 'K57.30', 10, 'Diverticulosis of colon without perforation or abscess without bleeding', 'Principal', TRUE),
('D30018', 'S20013', 'M17.9', 10, 'Osteoarthritis of knee, unspecified', 'Principal', TRUE),
('D30019', 'S20014', 'J18.9', 10, 'Pneumonia, unspecified organism', 'Principal', TRUE),
('D30020', 'S20015', 'K80.10', 10, 'Calculus of gallbladder with chronic cholecystitis without obstruction', 'Principal', TRUE),
('D30021', 'S20016', 'I48.91', 10, 'Unspecified atrial fibrillation', 'Principal', TRUE),
('D30022', 'S20017', 'S72.001A', 10, 'Fracture of unspecified part of neck of right femur, initial encounter', 'Principal', TRUE),
('D30023', 'S20018', 'I63.9', 10, 'Cerebral infarction, unspecified', 'Principal', TRUE),
('D30024', 'S20019', 'J45.901', 10, 'Unspecified asthma with (acute) exacerbation', 'Principal', TRUE),
('D30025', 'S20020', 'I50.9', 10, 'Heart failure, unspecified', 'Principal', TRUE);

INSERT INTO procedures 
(procedure_id, stay_id, procedure_code, procedure_date, description)
VALUES
('PR40001', 'S20001', '0JH60', '2023-01-06', 'Central venous catheter insertion'),
('PR40002', 'S20001', '5A1D00Z', '2023-01-07', 'Hemodialysis'),
('PR40003', 'S20002', 'BR31', '2023-01-12', 'Bronchoscopy with bronchoalveolar lavage'),
('PR40004', 'S20003', '3E033GC', '2023-01-16', 'Intravenous administration of thrombolytic agent'),
('PR40005', 'S20003', 'B020', '2023-01-17', 'Computed tomography (CT) scan of head'),
('PR40006', 'S20004', '0FT44ZZ', '2023-01-21', 'Laparoscopic cholecystectomy'),
('PR40007', 'S20005', '0SRC0J9', '2023-01-26', 'Total knee replacement'),
('PR40008', 'S20006', '0JH60', '2023-02-13', 'Central venous catheter insertion'),
('PR40009', 'S20007', '0HBT0ZZ', '2023-02-16', 'Excision of right breast lesion'),
('PR40010', 'S20008', '3E033GC', '2023-02-21', 'Intravenous administration of thrombolytic agent'),
('PR40011', 'S20009', '0DTJ0ZZ', '2023-02-26', 'Laparoscopic appendectomy'),
('PR40012', 'S20010', '0QS60', '2023-03-02', 'Hip fracture repair'),
('PR40013', 'S20011', 'BR31', '2023-03-06', 'Bronchoscopy with bronchoalveolar lavage'),
('PR40014', 'S20012', 'BF10', '2023-03-11', 'Colonoscopy with biopsy'),
('PR40015', 'S20013', '8E0W', '2023-03-16', 'Physical therapy'),
('PR40016', 'S20014', '3E0F3', '2023-03-21', 'Intravenous antibiotic administration'),
('PR40017', 'S20015', '0FT44ZZ', '2023-03-26', 'Laparoscopic cholecystectomy'),
('PR40018', 'S20016', '4A02X', '2023-04-03', 'Cardiac monitoring'),
('PR40019', 'S20017', '0QS60', '2023-04-06', 'Hip fracture repair'),
('PR40020', 'S20018', 'B020', '2023-04-11', 'Computed tomography (CT) scan of head'),
('PR40021', 'S20019', 'BR31', '2023-04-16', 'Bronchoscopy with bronchoalveolar lavage'),
('PR40022', 'S20020', '5A1D00Z', '2023-04-22', 'Hemodialysis');

INSERT INTO medications 
(medication_id, stay_id, medication_name, ndc_code, dosage, frequency, start_date, end_date)
VALUES
('M50001', 'S20001', 'Furosemide', '00071-0418-24', '40 mg', 'BID', '2023-01-05', '2023-01-12'),
('M50002', 'S20001', 'Lisinopril', '00071-0353-23', '10 mg', 'Daily', '2023-01-05', '2023-01-12'),
('M50003', 'S20001', 'Metformin', '00071-0861-23', '500 mg', 'BID', '2023-01-05', '2023-01-12'),
('M50004', 'S20002', 'Albuterol', '00173-0682-24', '2.5 mg', 'Q4H PRN', '2023-01-10', '2023-01-14'),
('M50005', 'S20002', 'Prednisone', '00054-8745-25', '40 mg', 'Daily', '2023-01-10', '2023-01-14'),
('M50006', 'S20003', 'Aspirin', '00904-2013-61', '81 mg', 'Daily', '2023-01-15', '2023-01-25'),
('M50007', 'S20003', 'Atorvastatin', '00071-0156-23', '40 mg', 'Bedtime', '2023-01-15', '2023-01-25'),
('M50008', 'S20003', 'Clopidogrel', '00071-0418-13', '75 mg', 'Daily', '2023-01-15', '2023-01-25'),
('M50009', 'S20004', 'Piperacillin-Tazobactam', '00071-1027-26', '4.5 g', 'Q6H', '2023-01-20', '2023-01-24'),
('M50010', 'S20005', 'Oxycodone', '00071-1025-23', '5 mg', 'Q4H PRN', '2023-01-25', '2023-02-02'),
('M50011', 'S20006', 'Furosemide', '00071-0418-24', '80 mg', 'BID', '2023-02-12', '2023-02-18'),
('M50012', 'S20006', 'Spironolactone', '00071-0418-32', '25 mg', 'Daily', '2023-02-12', '2023-02-18'),
('M50013', 'S20007', 'Morphine', '00121-0127-67', '2 mg', 'Q4H PRN', '2023-02-15', '2023-02-20'),
('M50014', 'S20008', 'Aspirin', '00904-2013-61', '81 mg', 'Daily', '2023-02-20', '2023-02-28'),
('M50015', 'S20008', 'Atorvastatin', '00071-0156-23', '40 mg', 'Bedtime', '2023-02-20', '2023-02-28'),
('M50016', 'S20009', 'Piperacillin-Tazobactam', '00071-1027-26', '4.5 g', 'Q6H', '2023-02-25', '2023-03-01'),
('M50017', 'S20010', 'Enoxaparin', '00071-1026-63', '40 mg', 'Daily', '2023-03-01', '2023-03-12'),
('M50018', 'S20011', 'Albuterol', '00173-0682-24', '2.5 mg', 'Q4H PRN', '2023-03-05', '2023-03-08'),
('M50019', 'S20012', 'Ciprofloxacin', '00093-1057-01', '500 mg', 'BID', '2023-03-10', '2023-03-15'),
('M50020', 'S20013', 'Acetaminophen', '00904-2013-60', '650 mg', 'Q6H PRN', '2023-03-15', '2023-03-20'),
('M50021', 'S20014', 'Ceftriaxone', '00071-1024-23', '1 g', 'Daily', '2023-03-20', '2023-03-28'),
('M50022', 'S20015', 'Ondansetron', '00071-0761-24', '4 mg', 'Q8H PRN', '2023-03-25', '2023-03-28'),
('M50023', 'S20016', 'Amiodarone', '00071-0237-23', '200 mg', 'BID', '2023-04-02', '2023-04-08'),
('M50024', 'S20017', 'Enoxaparin', '00071-1026-63', '40 mg', 'Daily', '2023-04-05', '2023-04-15'),
('M50025', 'S20018', 'Aspirin', '00904-2013-61', '81 mg', 'Daily', '2023-04-10', '2023-04-18'),
('M50026', 'S20019', 'Albuterol', '00173-0682-24', '2.5 mg', 'Q4H PRN', '2023-04-15', '2023-04-18'),
('M50027', 'S20020', 'Furosemide', '00071-0418-24', '80 mg', 'BID', '2023-04-20', '2023-04-30');

INSERT INTO lab_results 
(result_id, stay_id, test_name, test_date, result_value, unit, reference_range, abnormal_flag)
VALUES
('L60001', 'S20001', 'Sodium', '2023-01-06 08:30:00', '132', 'mmol/L', '135-145', 'L'),
('L60002', 'S20001', 'Potassium', '2023-01-06 08:30:00', '5.2', 'mmol/L', '3.5-5.0', 'H'),
('L60003', 'S20001', 'Creatinine', '2023-01-06 08:30:00', '1.8', 'mg/dL', '0.7-1.3', 'H'),
('L60004', 'S20001', 'BNP', '2023-01-06 09:15:00', '890', 'pg/mL', '<100', 'H'),
('L60005', 'S20001', 'Hemoglobin', '2023-01-06 08:30:00', '11.2', 'g/dL', '13.5-17.5', 'L'),
('L60006', 'S20002', 'Sodium', '2023-01-10 09:45:00', '138', 'mmol/L', '135-145', 'N'),
('L60007', 'S20002', 'Potassium', '2023-01-10 09:45:00', '4.1', 'mmol/L', '3.5-5.0', 'N'),
('L60008', 'S20002', 'Creatinine', '2023-01-10 09:45:00', '0.9', 'mg/dL', '0.7-1.3', 'N'),
('L60009', 'S20002', 'pO2', '2023-01-10 10:30:00', '65', 'mmHg', '75-100', 'L'),
('L60010', 'S20002', 'pCO2', '2023-01-10 10:30:00', '48', 'mmHg', '35-45', 'H'),
('L60011', 'S20003', 'Hemoglobin', '2023-01-15 11:00:00', '14.2', 'g/dL', '13.5-17.5', 'N'),
('L60012', 'S20003', 'Platelets', '2023-01-15 11:00:00', '210', '10^9/L', '150-450', 'N'),
('L60013', 'S20003', 'INR', '2023-01-15 11:00:00', '1.1', '', '0.8-1.2', 'N'),
('L60014', 'S20003', 'Troponin I', '2023-01-15 11:00:00', '0.01', 'ng/mL', '<0.04', 'N'),
('L60015', 'S20004', 'WBC', '2023-01-20 14:15:00', '12.5', '10^9/L', '4.5-11.0', 'H'),
('L60016', 'S20004', 'Total Bilirubin', '2023-01-20 14:15:00', '3.2', 'mg/dL', '0.1-1.2', 'H'),
('L60017', 'S20004', 'ALT', '2023-01-20 14:15:00', '85', 'U/L', '7-56', 'H'),
('L60018', 'S20004', 'AST', '2023-01-20 14:15:00', '90', 'U/L', '10-40', 'H'),
('L60019', 'S20005', 'Hemoglobin', '2023-01-25 08:45:00', '10.8', 'g/dL', '13.5-17.5', 'L'),
('L60020', 'S20005', 'Hematocrit', '2023-01-25 08:45:00', '32.4', '%', '41-53', 'L'),
('L60021', 'S20006', 'Sodium', '2023-02-12 10:30:00', '130', 'mmol/L', '135-145', 'L'),
('L60022', 'S20006', 'Potassium', '2023-02-12 10:30:00', '5.5', 'mmol/L', '3.5-5.0', 'H'),
('L60023', 'S20006', 'Creatinine', '2023-02-12 10:30:00', '2.1', 'mg/dL', '0.7-1.3', 'H'),
('L60024', 'S20006', 'BNP', '2023-02-12 11:15:00', '1200', 'pg/mL', '<100', 'H'),
('L60025', 'S20007', 'Hemoglobin', '2023-02-15 09:00:00', '9.8', 'g/dL', '12.0-15.5', 'L');

INSERT INTO comorbidities 
(comorbidity_id, patient_id, condition_name, icd_code, diagnosed_date)
VALUES
('C70001', 'P10001', 'Hypertension', 'I10', '2018-05-15'),
('C70002', 'P10001', 'Type 2 Diabetes', 'E11.9', '2019-08-10'),
('C70003', 'P10001', 'Chronic Kidney Disease, Stage 3', 'N18.3', '2021-03-20'),
('C70004', 'P10002', 'Asthma', 'J45.909', '2010-11-05'),
('C70005', 'P10002', 'Chronic Obstructive Pulmonary Disease', 'J44.9', '2020-02-18'),
('C70006', 'P10003', 'Hypertension', 'I10', '2015-07-22'),
('C70007', 'P10003', 'Atrial Fibrillation', 'I48.91', '2018-04-30'),
('C70008', 'P10003', 'Hyperlipidemia', 'E78.5', '2016-09-15'),
('C70009', 'P10004', 'Cholelithiasis', 'K80.20', '2022-08-12'),
('C70010', 'P10005', 'Osteoarthritis, knee', 'M17.9', '2019-03-05'),
('C70011', 'P10005', 'Hypertension', 'I10', '2017-01-20'),
('C70012', 'P10006', 'Breast Cancer', 'C50.911', '2022-09-15'),
('C70013', 'P10006', 'Depression', 'F32.9', '2020-06-10'),
('C70014', 'P10007', 'Appendicitis', 'K35.80', '2023-02-24'),
('C70015', 'P10008', 'Osteoporosis', 'M81.0', '2018-11-30'),
('C70016', 'P10008', 'Hypertension', 'I10', '2016-02-15'),
('C70017', 'P10009', 'Diverticulosis', 'K57.30', '2021-07-18'),
('C70018', 'P10010', 'Hypertension', 'I10', '2015-05-10'),
('C70019', 'P10010', 'Congestive Heart Failure', 'I50.9', '2022-01-15'),
('C70020', 'P10010', 'Type 2 Diabetes', 'E11.9', '2016-08-22');