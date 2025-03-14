# Diabetes-Specific Analysis
# This code should be added to the exploratory_data_analysis.ipynb notebook

# Check if we have glucose and A1C test data
glucose_cols = [col for col in df.columns if 'glucose' in col.lower()]
a1c_cols = [col for col in df.columns if 'a1c' in col.lower()]

if glucose_cols and a1c_cols:
    print("# Diabetes Analysis\n")
    
    glucose_col = glucose_cols[0]  # Use the first matching column
    if df[glucose_col].nunique() > 0:
        plt.figure(figsize=(12, 6))
        
        glucose_counts = df[glucose_col].value_counts().sort_index()
        ax = sns.barplot(x=glucose_counts.index, y=glucose_counts.values)
        plt.title('Distribution of Glucose Test Results', fontsize=14)
        plt.xlabel('Glucose Test Result', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        for i, v in enumerate(glucose_counts.values):
            ax.text(i, v + 5, str(v), ha='center')
        
        plt.tight_layout()
        plt.show()
        
        if readmission_col:
            plt.figure(figsize=(12, 6))
            readmission_by_glucose = df.groupby(glucose_col)[readmission_col].mean() * 100
            
            ax = sns.barplot(x=readmission_by_glucose.index, y=readmission_by_glucose.values, palette='viridis')
            plt.title(f'Readmission Rate by Glucose Test Result', fontsize=14)
            plt.xlabel('Glucose Test Result', fontsize=12)
            plt.ylabel('Readmission Rate (%)', fontsize=12)
            
            for i, v in enumerate(readmission_by_glucose.values):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center')
            
            plt.tight_layout()
            plt.show()
    
    a1c_col = a1c_cols[0]  # Use the first matching column
    if df[a1c_col].nunique() > 0:
        plt.figure(figsize=(12, 6))
        
        a1c_counts = df[a1c_col].value_counts().sort_index()
        ax = sns.barplot(x=a1c_counts.index, y=a1c_counts.values)
        plt.title('Distribution of A1C Test Results', fontsize=14)
        plt.xlabel('A1C Test Result', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        for i, v in enumerate(a1c_counts.values):
            ax.text(i, v + 5, str(v), ha='center')
        
        plt.tight_layout()
        plt.show()
        
        if readmission_col:
            plt.figure(figsize=(12, 6))
            readmission_by_a1c = df.groupby(a1c_col)[readmission_col].mean() * 100
            
            ax = sns.barplot(x=readmission_by_a1c.index, y=readmission_by_a1c.values, palette='viridis')
            plt.title(f'Readmission Rate by A1C Test Result', fontsize=14)
            plt.xlabel('A1C Test Result', fontsize=12)
            plt.ylabel('Readmission Rate (%)', fontsize=12)
            
            for i, v in enumerate(readmission_by_a1c.values):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center')
            
            plt.tight_layout()
            plt.show()
    
    medication_cols = [col for col in df.columns if 'medication' in col.lower() or 'med' in col.lower()]
    change_cols = [col for col in df.columns if 'change' in col.lower()]
    
    if medication_cols and change_cols:
        medication_col = medication_cols[0]
        change_col = change_cols[0]
        
        plt.figure(figsize=(12, 6))
        cross_tab = pd.crosstab(df[medication_col], df[change_col])
        cross_tab.plot(kind='bar', stacked=True)
        plt.title('Medication Changes by Diabetes Medication Status', fontsize=14)
        plt.xlabel('Diabetes Medication', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(title='Medication Change')
        plt.tight_layout()
        plt.show()
        
        if readmission_col:
            plt.figure(figsize=(12, 6))
            pivot_data = df.pivot_table(
                index=medication_col, 
                columns=change_col,
                values=readmission_col,
                aggfunc='mean'
            ) * 100
            
            pivot_data.plot(kind='bar')
            plt.title('Readmission Rate by Medication Status and Change', fontsize=14)
            plt.xlabel('Diabetes Medication', fontsize=12)
            plt.ylabel('Readmission Rate (%)', fontsize=12)
            plt.legend(title='Medication Change')
            plt.tight_layout()
            plt.show()
    
    if glucose_cols and medication_cols and readmission_col:
        plt.figure(figsize=(14, 8))
        
        df['glucose_med_combined'] = df[glucose_col] + '_' + df[medication_col]
        
        combined_readmission = df.groupby('glucose_med_combined')[readmission_col].agg(['mean', 'count'])
        combined_readmission['mean'] = combined_readmission['mean'] * 100
        
        combined_readmission = combined_readmission[combined_readmission['count'] >= 10]
        
        combined_readmission = combined_readmission.sort_values('mean', ascending=False)
        
        # Plot
        ax = sns.barplot(x=combined_readmission.index, y=combined_readmission['mean'], palette='viridis')
        plt.title('Readmission Rate by Glucose Result and Medication Status', fontsize=14)
        plt.xlabel('Glucose Result _ Medication Status', fontsize=12)
        plt.ylabel('Readmission Rate (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        for i, (rate, count) in enumerate(zip(combined_readmission['mean'], combined_readmission['count'])):
            ax.text(i, rate + 1, f'{rate:.1f}%\n(n={count})', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()

drug_cols = [col for col in df.columns if any(term in col.lower() for term in ['drug', 'medication', 'effective', 'satisfaction'])]

if len(drug_cols) >= 3:  # If we have at least a few drug-related columns
    print("# Drug Performance Analysis\n")
    
    effect_col = next((col for col in drug_cols if 'effect' in col.lower()), None)
    satis_col = next((col for col in drug_cols if 'satis' in col.lower()), None)
    drug_col = next((col for col in drug_cols if 'drug' in col.lower() or 'medication' in col.lower()), None)
    
    if effect_col and drug_col:
        plt.figure(figsize=(14, 8))
        
        top_drugs = df[drug_col].value_counts().head(10).index
        
        drug_effect = df[df[drug_col].isin(top_drugs)].groupby(drug_col)[effect_col].mean().sort_values(ascending=False)
        
        ax = sns.barplot(x=drug_effect.index, y=drug_effect.values, palette='viridis')
        plt.title('Drug Effectiveness Ratings (Top 10 Drugs)', fontsize=14)
        plt.xlabel('Drug', fontsize=12)
        plt.ylabel('Average Effectiveness Rating', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        for i, v in enumerate(drug_effect.values):
            ax.text(i, v + 0.1, f'{v:.1f}', ha='center')
        
        plt.tight_layout()
        plt.show()
    
    if effect_col and readmission_col:
        plt.figure(figsize=(12, 6))
        
        df['effect_bin'] = pd.cut(df[effect_col], bins=5)
        
        effect_readmission = df.groupby('effect_bin')[readmission_col].agg(['mean', 'count'])
        effect_readmission['mean'] = effect_readmission['mean'] * 100
        
        
        ax = sns.barplot(x=effect_readmission.index.astype(str), y=effect_readmission['mean'], palette='viridis')
        plt.title('Readmission Rate by Drug Effectiveness', fontsize=14)
        plt.xlabel('Drug Effectiveness Rating (Binned)', fontsize=12)
        plt.ylabel('Readmission Rate (%)', fontsize=12)
        
        for i, (rate, count) in enumerate(zip(effect_readmission['mean'], effect_readmission['count'])):
            ax.text(i, rate + 1, f'{rate:.1f}%\n(n={count})', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
