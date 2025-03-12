

Create table Patients(
    primary key ID INT,
    name varchar(255),
    Age INT,
    Gender varchar(255),
    Diagnosis varchar(255),
    Treatment varchar(255),
    Readmission varchar(255),
    Insurance varchar (255)
);

Create Table Admissions (
    Primary Key Patient_ID INT,
    Admission_Date datetime,
    Discharge_date datetime,
    Length_of_stay INT
);

Create Table Medications (
    Primary Key Patient_ID INT,
    Drug_Name varchar(255),
    Dosage INT,
    Tearment_Effectivness INT
);

