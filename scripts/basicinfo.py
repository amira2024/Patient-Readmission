

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

dfHD = pd.read_csv("data/raw/healthcare_dataset.csv")

print("Dataset Shape:", dfHD.shape) #Returns a tuple (how many rows, how many columns)
print("\nData Types:") #Data types of each column
print(dfHD.dtypes)

print("\nMissing Values:")
print(dfHD.isnull().sum())


print("\nSummary Statistics:")
print(dfHD.describe())

print("\nSample Data:")
print(dfHD.head())

dfHD['Date of Admission'] = pd.to_datetime(dfHD['Date of Admission'])
dfHD['Discharge Date'] = pd.to_datetime(dfHD['Discharge Date'])


dfHD['Length of Stay'] = (dfHD['Discharge Date'] - dfHD['Date of Admission']).dt.days

plt.figure(figsize=(10, 6))
sns.histplot(dfHD['Length of Stay'])
plt.title('Distribution of Length of Stay')
plt.xlabel('Days')
plt.savefig('length_of_stay_distribution.png')


plt.figure(figsize=(12, 8))
condition_counts = dfHD['Medical Condition'].value_counts().head(10)
sns.barplot(x=condition_counts.values, y=condition_counts.index)
plt.title('Top 10 Medical Conditions')
plt.xlabel('Count')
plt.savefig('top_medical_conditions.png')
