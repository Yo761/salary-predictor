"""
Feature Engineering Module
Converts raw data into ML-ready features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class FeatureEngineer:
    """Transforms raw job data into ML features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=30)
        self.location_salary_map = None
        self.company_size_map = None
    
    def fit(self, df):
        """Learn from training data"""
        
        # Location encoding based on median salary
        self.location_salary_map = df.groupby('location')['salary'].median().to_dict()
        
        # Company size ordering
        self.company_size_map = {
            'Startup': 1, 'Small': 2, 'Mid': 3, 'Large': 4, 'Enterprise': 5
        }
        
        # Fit label encoders for categorical variables
        for col in ['job_title', 'company', 'industry', 'education']:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Fit TF-IDF on skills
        self.tfidf_vectorizer.fit(df['skills'])
        
        return self
    
    def transform(self, df):
        """Apply transformations"""
        df = df.copy()
        
        # ===== NUMERICAL FEATURES =====
        df['experience_squared'] = df['experience_required'] ** 2
        df['experience_log'] = np.log1p(df['experience_required'])
        
        # ===== CATEGORICAL ENCODING =====
        # Location: encode by median salary (higher salary cities get higher value)
        df['location_encoded'] = df['location'].map(self.location_salary_map)
        
        # Company size: ordinal encoding
        df['company_size_encoded'] = df['company_size'].map(self.company_size_map)
        
        # Job title, company, industry, education: label encoding
        for col in ['job_title', 'company', 'industry', 'education']:
            df[col + '_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
        
        # ===== TEXT FEATURES (Skills) =====
        skills_tfidf = self.tfidf_vectorizer.transform(df['skills']).toarray()
        skills_df = pd.DataFrame(
            skills_tfidf,
            columns=[f'skill_{i}' for i in range(skills_tfidf.shape[1])]
        )
        df = pd.concat([df, skills_df], axis=1)
        
        # ===== TEMPORAL FEATURES =====
        df['posted_date'] = pd.to_datetime(df['posted_date'])
        df['posted_days_ago'] = (pd.Timestamp.now() - df['posted_date']).dt.days
        df['posted_month'] = df['posted_date'].dt.month
        
        # ===== JOB TITLE POPULARITY =====
        job_title_counts = df['job_title'].value_counts()
        df['job_title_popularity'] = df['job_title'].map(job_title_counts)
        
        # ===== INTERACTION FEATURES =====
        df['seniority_score'] = df['experience_required'] * df['company_size_encoded']
        df['location_experience_interaction'] = df['location_encoded'] * df['experience_required']
        
        return df
    
    def get_feature_columns(self):
        """Return list of feature columns for modeling"""
        feature_cols = [
            'experience_required', 'experience_squared', 'experience_log',
            'location_encoded', 'company_size_encoded',
            'job_title_encoded', 'company_encoded', 'industry_encoded',
            'education_encoded', 'job_title_popularity',
            'posted_days_ago', 'posted_month',
            'seniority_score', 'location_experience_interaction'
        ]
        
        # Add skill features
        skill_cols = [f'skill_{i}' for i in range(30)]
        feature_cols.extend(skill_cols)
        
        return feature_cols
    
    def save(self, filepath='../models/preprocessor.pkl'):
        """Save fitted preprocessor"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath='../models/preprocessor.pkl'):
        """Load fitted preprocessor"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def preprocess_data():
    """Main preprocessing pipeline"""
    
    print("📂 Loading raw data...")
    df = pd.read_csv('data/raw/jobs_raw.csv')
    print(f"✅ Loaded {len(df)} records")
    
    # Check for missing values
    print("\n📊 Missing values:")
    print(df.isnull().sum())
    
    # Remove rows with missing critical columns
    df = df.dropna(subset=['salary', 'job_title', 'location'])
    print(f"✅ After removing nulls: {len(df)} records")
    
    # Remove salary outliers (keep 1st-99th percentile)
    q1 = df['salary'].quantile(0.01)
    q99 = df['salary'].quantile(0.99)
    df = df[(df['salary'] >= q1) & (df['salary'] <= q99)]
    print(f"✅ After removing outliers: {len(df)} records")
    
    # Fit and transform
    print("\n🔧 Feature engineering...")
    engineer = FeatureEngineer()
    engineer.fit(df)
    df_transformed = engineer.transform(df)
    
    # Get feature columns
    feature_cols = engineer.get_feature_columns()
    
    # Prepare X (features) and y (target)
    X = df_transformed[feature_cols]
    y = df_transformed['salary']
    
    print(f"✅ Created {X.shape[1]} features")
    print(f"✅ Target variable shape: {y.shape}")
    
    # Save processed data
    df_transformed.to_csv('data/processed/jobs_processed.csv', index=False)
    X.to_csv('data/processed/X_features.csv', index=False)
    y.to_csv('data/processed/y_target.csv', index=False)
    
    # Save preprocessor
    os.makedirs('models', exist_ok=True)
    engineer.save('models/preprocessor.pkl')
    
    print("\n✅ Preprocessing complete!")
    print("📁 Saved:")
    print("  - data/processed/jobs_processed.csv")
    print("  - data/processed/X_features.csv")
    print("  - data/processed/y_target.csv")
    print("  - models/preprocessor.pkl")
    
    return X, y, engineer


if __name__ == "__main__":
    X, y, engineer = preprocess_data()