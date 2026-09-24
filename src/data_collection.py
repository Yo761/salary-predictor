"""
Data Collection Module
Generates realistic Indian job market dataset
In production, this would use web scraping (BeautifulSoup, Scrapy)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class JobDataGenerator:
    """Generate realistic Indian job market data"""
    
    def __init__(self, n_records=10000, seed=42):
        self.n_records = n_records
        np.random.seed(seed)
        random.seed(seed)
        
        # Define realistic options for Indian job market
        self.job_titles = [
            'Data Scientist', 'Machine Learning Engineer', 'Data Engineer',
            'Python Developer', 'Full Stack Developer', 'Backend Engineer',
            'Frontend Developer', 'DevOps Engineer', 'Cloud Engineer',
            'AI/ML Engineer', 'Analytics Engineer', 'Business Analyst',
            'Product Manager', 'Software Engineer', 'Senior Developer',
            'Junior Developer', 'Solutions Architect', 'Technical Lead'
        ]
        
        self.locations = [
            'Bangalore', 'Hyderabad', 'Delhi', 'Mumbai', 'Pune',
            'Chennai', 'Kolkata', 'Gurugram', 'Noida', 'Jaipur'
        ]
        
        self.companies = [
            'Google', 'Microsoft', 'Amazon', 'Meta', 'Apple',
            'Flipkart', 'OYO', 'Paytm', 'Unacademy', 'Swiggy',
            'Airbnb', 'Uber', 'IBM', 'TCS', 'Infosys',
            'Wipro', 'HCL', 'Accenture', 'Cognizant', 'Capgemini',
            'Startup_A', 'Startup_B', 'Startup_C', 'Startup_D', 'Startup_E'
        ]
        
        self.company_sizes = ['Startup', 'Small', 'Mid', 'Large', 'Enterprise']
        
        self.industries = [
            'Technology', 'Finance', 'E-commerce', 'Healthcare',
            'Education', 'Consulting', 'Manufacturing', 'Logistics'
        ]
        
        self.skills_pool = [
            'Python', 'Java', 'SQL', 'Machine Learning', 'Deep Learning',
            'TensorFlow', 'PyTorch', 'Pandas', 'NumPy', 'Scikit-learn',
            'AWS', 'GCP', 'Azure', 'Docker', 'Kubernetes',
            'Spark', 'Hadoop', 'Kafka', 'PostgreSQL', 'MongoDB',
            'JavaScript', 'React', 'Node.js', 'Angular', 'Django',
            'FastAPI', 'REST API', 'GraphQL', 'Git', 'Linux',
            'Statistics', 'R', 'Tableau', 'Power BI', 'Excel',
            'Agile', 'JIRA', 'Communication', 'Leadership', 'Problem Solving'
        ]
        
        self.education = ['Bachelor', 'Master', 'PhD', 'Diploma', 'Bootcamp']
    
    def generate_salary(self, job_title, experience, location, company_size, skills_count):
        """
        Generate realistic salary based on Indian market
        Base salary logic:
        - Entry level: 5L - 8L INR
        - Mid-level (3-5 yrs): 10L - 15L INR
        - Senior (8+ yrs): 20L - 40L INR
        """
        
        # Base salary by job title
        job_base = {
            'Data Scientist': 8.5,
            'Machine Learning Engineer': 8.5,
            'Data Engineer': 8.0,
            'Senior Developer': 12.0,
            'Python Developer': 6.5,
            'Full Stack Developer': 7.0,
            'Backend Engineer': 7.5,
            'Frontend Developer': 6.5,
            'DevOps Engineer': 8.5,
            'Cloud Engineer': 8.0,
            'AI/ML Engineer': 9.0,
            'Analytics Engineer': 7.0,
            'Software Engineer': 7.0,
            'Junior Developer': 4.5,
            'Technical Lead': 13.0,
            'Solutions Architect': 12.0,
            'Product Manager': 11.0,
            'Business Analyst': 6.0
        }
        
        # Location multiplier (Tier-1 cities pay more)
        location_multiplier = {
            'Bangalore': 1.15,
            'Delhi': 1.10,
            'Mumbai': 1.12,
            'Hyderabad': 1.08,
            'Pune': 1.05,
            'Gurugram': 1.10,
            'Noida': 1.05,
            'Chennai': 0.95,
            'Kolkata': 0.90,
            'Jaipur': 0.85
        }
        
        # Company size multiplier
        size_multiplier = {
            'Enterprise': 1.20,
            'Large': 1.12,
            'Mid': 1.0,
            'Small': 0.90,
            'Startup': 0.80
        }
        
        # Experience multiplier (exponential growth)
        exp_multiplier = 1 + (experience * 0.08)  # 8% increase per year
        
        # Skills multiplier (more skills = higher pay)
        skills_multiplier = 1 + (min(skills_count, 10) * 0.03)
        
        base = job_base.get(job_title, 7.0)
        
        salary = (base * location_multiplier.get(location, 1.0) * 
                 size_multiplier.get(company_size, 1.0) * 
                 exp_multiplier * skills_multiplier)
        
        # Add random noise (±10%)
        noise = np.random.normal(1, 0.10)
        salary = salary * noise
        
        return max(salary, 3.0)  # Minimum 3L
    
    def generate(self):
        """Generate the complete dataset"""
        
        data = {
            'job_id': [f'JOB_{i:06d}' for i in range(self.n_records)],
            'job_title': np.random.choice(self.job_titles, self.n_records),
            'company': np.random.choice(self.companies, self.n_records),
            'location': np.random.choice(self.locations, self.n_records),
            'company_size': np.random.choice(self.company_sizes, self.n_records),
            'industry': np.random.choice(self.industries, self.n_records),
            'experience_required': np.random.choice(range(0, 15), self.n_records),
            'education': np.random.choice(self.education, self.n_records),
            'posted_date': [
                datetime.now() - timedelta(days=random.randint(1, 365))
                for _ in range(self.n_records)
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Generate skills (2-8 skills per job)
        df['skills'] = [
            ','.join(np.random.choice(self.skills_pool, random.randint(2, 8), replace=False))
            for _ in range(self.n_records)
        ]
        
        # Generate salary using our logic
        df['salary'] = df.apply(
            lambda row: self.generate_salary(
                row['job_title'],
                row['experience_required'],
                row['location'],
                row['company_size'],
                len(row['skills'].split(','))
            ),
            axis=1
        )
        
        # Convert salary to lakhs (L) for Indian market (common denomination)
        df['salary'] = df['salary'].round(2)
        
        # Add engagement metrics
        df['applications'] = np.random.choice(range(10, 500), self.n_records)
        df['views'] = np.random.choice(range(100, 5000), self.n_records)
        
        return df


def scrape_real_data():
    """
    In production, this would actually scrape LinkedIn/Indeed
    Example structure (not actually running):
    
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException
    
    def scrape_indeed():
        driver = webdriver.Chrome()
        driver.get('https://www.indeed.co.in/')
        
        # Fill search criteria
        search_input = driver.find_element_by_id("text-input-what")
        search_input.send_keys("Data Scientist")
        
        location_input = driver.find_element_by_id("text-input-where")
        location_input.send_keys("Bangalore")
        
        # Parse results
        jobs = []
        for job in driver.find_elements_by_class_name("job_seen_beacon"):
            title = job.find_element_by_class_name("jobTitle").text
            salary = job.find_element_by_class_name("salary").text
            # ... parse more fields
            jobs.append({...})
        
        return jobs
    """
    pass


if __name__ == "__main__":
    print("🔄 Generating Indian Job Market Dataset...")
    
    generator = JobDataGenerator(n_records=10000)
    df = generator.generate()
    
    # Save raw data
    df.to_csv('data/raw/jobs_raw.csv', index=False)
    print(f"✅ Generated {len(df)} job records")
    print(f"📊 Data shape: {df.shape}")
    print(f"📁 Saved to: data/raw/jobs_raw.csv")
    
    print("\n📋 Dataset Preview:")
    print(df.head(10))
    
    print("\n📈 Basic Statistics:")
    print(df[['experience_required', 'salary', 'applications']].describe())