def process_drug_performance_data(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the drug performance dataset and prepare it for integration
    with the readmission analysis.
    
    Args:
        df: Raw drug performance DataFrame
        
    Returns:
        Processed DataFrame ready for analysis
    """
    logger.info("Processing drug performance data")
    
    drug_df = df.copy()
    
    expected_columns = ['Drug', 'Condition', 'Type', 'Effective', 'Satisfaction']
    missing_columns = [col for col in expected_columns if col not in drug_df.columns]
    
    if missing_columns:
        logger.warning(f"Missing expected columns in drug performance data: {missing_columns}")
        for col in missing_columns:
            drug_df[col] = None
    
    drug_df['Drug'] = drug_df['Drug'].fillna('Unknown')
    drug_df['Condition'] = drug_df['Condition'].fillna('Unknown')
    drug_df['Type'] = drug_df['Type'].fillna('Unknown')
    
    numeric_cols = ['Effective', 'EaseOfUse', 'Satisfaction', 'Reviews']
    for col in numeric_cols:
        if col in drug_df.columns:
            drug_df[col] = pd.to_numeric(drug_df[col], errors='coerce')
            drug_df[col] = drug_df[col].fillna(drug_df[col].median())
    
    for col in ['Effective', 'EaseOfUse', 'Satisfaction']:
        if col in drug_df.columns:
            max_val = drug_df[col].max()
            if max_val <= 10:  # Assuming a 1-10 scale
                drug_df[col] = (drug_df[col] / 10) * 100
            elif max_val <= 5:  # Assuming a 1-5 scale
                drug_df[col] = (drug_df[col] / 5) * 100
    
    if all(col in drug_df.columns for col in ['Effective', 'EaseOfUse', 'Satisfaction']):
        drug_df['composite_score'] = (
            drug_df['Effective'] * 0.5 + 
            drug_df['EaseOfUse'] * 0.2 + 
            drug_df['Satisfaction'] * 0.3
        )
    
    # Create drug categories based on conditions
    if 'Condition' in drug_df.columns:
        drug_df['condition_category'] = drug_df['Condition'].str.split(',').str[0]
        common_conditions = {
            'diabetes': ['diabetes', 'type 2', 'type 1', 'blood sugar'],
            'heart': ['heart', 'cardiac', 'cardiovascular'],
            'respiratory': ['asthma', 'copd', 'respiratory', 'lung'],
            'pain': ['pain', 'arthritis', 'inflammation']
        }
        
        for category, keywords in common_conditions.items():
            mask = drug_df['Condition'].str.lower().str.contains('|'.join(keywords), na=False)
            drug_df.loc[mask, 'condition_category'] = category
    
    # Group drugs by generic name if available
    if 'Drug' in drug_df.columns and 'Type' in drug_df.columns:
        drug_df['generic_name'] = drug_df['Drug'].str.lower()
        
        brand_indicators = [' xr', ' er', ' sr', ' cr', ' ir', ' dr', ' xl', ' hct']
        for indicator in brand_indicators:
            drug_df['generic_name'] = drug_df['generic_name'].str.replace(indicator, '', regex=False)
        
        # For generic drugs, use their name as is
        drug_df.loc[drug_df['Type'].str.lower() == 'generic', 'generic_name'] = drug_df.loc[drug_df['Type'].str.lower() == 'generic', 'Drug'].str.lower()
    
    if 'generic_name' in drug_df.columns and 'Effective' in drug_df.columns:
        drug_effectiveness = drug_df.groupby('generic_name')['Effective'].mean().to_dict()
        
        drug_df['avg_drug_effectiveness'] = drug_df['generic_name'].map(drug_effectiveness)
    
    logger.info(f"Processed {len(drug_df)} drug performance records")
    logger.info(f"Number of unique drugs: {drug_df['Drug'].nunique()}")
    if 'condition_category' in drug_df.columns:
        logger.info(f"Top condition categories: {drug_df['condition_category'].value_counts().head(5)}")
    
    return drug_df
