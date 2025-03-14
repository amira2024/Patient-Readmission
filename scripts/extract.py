
import pandas as pd
import requests
import json
import os

def extract_csv_data(filepath):
    """Extract data from CSV files"""
    return pd.read_csv(filepath)

def extract_api_data(api_url, params=None):
    """Extract data from CMS API"""
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching API data: {response.status_code}")
        return None

def main():
    # Create data directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    patient_data = extract_csv_data("data/raw/healthcare_dataset.csv")
    readmissions_data = extract_csv_data("data/raw/to/hospital_readmissions.csv")
    drug_data = extract_csv_data("data/raw/Drug.csv")
    
    patient_data.to_csv("data/raw/raw_patient_data.csv", index=False)
    readmissions_data.to_csv("data/raw/raw_readmissions_data.csv", index=False)
    drug_data.to_csv("data/raw/raw_drug_data.csv", index=False)
    
    api_url = "​/provider-data​/api​/1​/metastore​/schemas​/dataset​/items​/9n3s-kdb3"
    params = {
        "query": "SELECT * FROM `9n3s-kdb3` LIMIT 1000"
    }
    cms_data = extract_api_data(api_url, params)
    
    if cms_data:
        with open("data/raw/raw_cms_data.json", "w") as f:
            json.dump(cms_data, f)

if __name__ == "__main__":
    main()


