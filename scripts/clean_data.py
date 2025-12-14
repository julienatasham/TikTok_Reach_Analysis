import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self, df):
        """
        Initialize DataCleaner with dataframe
        """
        self.df = df.copy()
        self.original_shape = df.shape
        self.original_columns = list(df.columns)
        
    def clean_data(self, save_to_processed=True, filename="cleaned_tiktok_data.csv"):
        """
        Perform all cleaning steps
        """
        print(" Starting data cleaning...")
        
        # 1. Check for duplicates
        self.remove_duplicates()
        
        # 2. Handle missing values
        self.handle_missing_values()
        
        # 3. Convert data types
        self.convert_data_types()
        
        # 4. Handle outliers
        self.handle_outliers()
        
        # 5. Create new features
        self.create_features()
        
        print(f"\n Data cleaning complete!")
        print(f"   Original shape: {self.original_shape}")
        print(f"   Final shape: {self.df.shape}")
        
        # Save to processed folder
        if save_to_processed:
            self.save_to_processed_folder(filename)
        
        return self.df
    
    def remove_duplicates(self):
        """
        Remove duplicate rows
        """
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"   Removing {duplicates} duplicate rows")
            self.df = self.df.drop_duplicates()
        else:
            print(f"   No duplicates found")
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset
        """
        missing = self.df.isnull().sum()
        total_missing = missing.sum()
        
        if total_missing > 0:
            print(f"   Found {total_missing} missing values")
            for col, count in missing[missing > 0].items():
                percentage = (count / len(self.df)) * 100
                print(f"     - {col}: {count} missing ({percentage:.1f}%)")
            
            # For numerical columns, fill with median
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                    print(f"     Filled {col} with median")
            
            # For categorical columns, fill with mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                    print(f"     Filled {col} with mode")
        else:
            print(f"   No missing values to handle")
    
    def convert_data_types(self):
        """
        Convert columns to appropriate data types
        """
        changes_made = False
        
        # Convert Date_Posted to datetime
        if 'Date_Posted' in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df['Date_Posted']):
                self.df['Date_Posted'] = pd.to_datetime(self.df['Date_Posted'])
                print(f"   Converted Date_Posted to datetime")
                changes_made = True
        
        # Don't convert Post_Hour to categorical yet - we need it as integer for comparisons
        # We'll convert it later after creating features
        if not changes_made:
            print(f"   No data type conversions needed")
    
    def handle_outliers(self):
        """
        Handle outliers in numerical columns using IQR method
        """
        numerical_cols = ['Reach', 'Likes', 'Comments', 'Shares', 
                         'Duration_Seconds', 'Caption_Length']
        
        outlier_count = 0
        for col in numerical_cols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Count outliers
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                outlier_count += outliers
                
                # Cap outliers instead of removing
                self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                
                if outliers > 0:
                    print(f"   Capped {outliers} outliers in {col}")
        
        if outlier_count > 0:
            print(f"   Capped {outlier_count} total outliers")
        else:
            print(f"   No significant outliers found")
    
    def create_features(self):
        """
        Create new features for analysis
        """
        print(f"\n Creating new features...")
        original_col_count = len(self.df.columns)
        
        # Engagement metrics
        self.df['Total_Engagement'] = self.df['Likes'] + self.df['Comments'] + self.df['Shares']
        self.df['Engagement_Rate'] = self.df['Total_Engagement'] / self.df['Reach']
        self.df['Like_Ratio'] = self.df['Likes'] / self.df['Reach']
        self.df['Comment_Ratio'] = self.df['Comments'] / self.df['Reach']
        self.df['Share_Ratio'] = self.df['Shares'] / self.df['Reach']
        print(f"   Created engagement metrics")
        
        # Date features
        if 'Date_Posted' in self.df.columns:
            self.df['Year'] = self.df['Date_Posted'].dt.year
            self.df['Month'] = self.df['Date_Posted'].dt.month
            self.df['Day'] = self.df['Date_Posted'].dt.day
            self.df['DayOfWeek'] = self.df['Date_Posted'].dt.dayofweek
            self.df['Weekend'] = self.df['DayOfWeek'].isin([5, 6]).astype(int)
            self.df['Quarter'] = self.df['Date_Posted'].dt.quarter
            self.df['WeekOfYear'] = self.df['Date_Posted'].dt.isocalendar().week
            print(f"   Created date features")
        
        # Time categories - FIXED: Keep Post_Hour as integer for comparisons
        if 'Post_Hour' in self.df.columns:
            # First ensure Post_Hour is integer
            if not pd.api.types.is_integer_dtype(self.df['Post_Hour']):
                self.df['Post_Hour'] = self.df['Post_Hour'].astype(int)
            
            # Create time categories
            self.df['Time_Category'] = pd.cut(
                self.df['Post_Hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                include_lowest=True
            )
            
            # Create peak hours indicator (8 AM - 10 PM)
            self.df['Peak_Hours'] = ((self.df['Post_Hour'] >= 8) & (self.df['Post_Hour'] <= 22)).astype(int)
            
            # Now convert to ordered categorical for better analysis
            self.df['Post_Hour_Category'] = pd.Categorical(
                self.df['Post_Hour'].astype(str) + ':00',
                categories=[f"{i:02d}:00" for i in range(24)],
                ordered=True
            )
            
            print(f"   Created time categories")
        
        # Content type based on duration
        if 'Duration_Seconds' in self.df.columns:
            self.df['Content_Type'] = pd.cut(
                self.df['Duration_Seconds'],
                bins=[0, 15, 30, 60, 120, float('inf')],
                labels=['Very_Short', 'Short', 'Medium', 'Long', 'Very_Long']
            )
            print(f"   Created content type categories")
        
        # Hashtag categories
        if 'Hashtags_Count' in self.df.columns:
            self.df['Hashtag_Category'] = pd.cut(
                self.df['Hashtags_Count'],
                bins=[0, 3, 6, 10, float('inf')],
                labels=['Few', 'Moderate', 'Many', 'Excessive']
            )
            print(f"   Created hashtag categories")
        
        # Caption length categories
        if 'Caption_Length' in self.df.columns:
            self.df['Caption_Category'] = pd.cut(
                self.df['Caption_Length'],
                bins=[0, 20, 50, 100, float('inf')],
                labels=['Short', 'Medium', 'Long', 'Very_Long']
            )
            print(f"   Created caption categories")
        
        # Performance categories
        self.df['Reach_Category'] = pd.qcut(self.df['Reach'], q=4, 
                                           labels=['Low', 'Medium', 'High', 'Very_High'])
        self.df['Engagement_Category'] = pd.qcut(self.df['Total_Engagement'], q=4,
                                                labels=['Low', 'Medium', 'High', 'Very_High'])
        print(f"   Created performance categories")
        
        # Interaction ratios
        self.df['Comments_Per_Like'] = self.df['Comments'] / self.df['Likes'].replace(0, 1)
        self.df['Shares_Per_Like'] = self.df['Shares'] / self.df['Likes'].replace(0, 1)
        self.df['Virality_Score'] = (self.df['Shares'] * 2 + self.df['Comments']) / self.df['Likes'].replace(0, 1)
        print(f"   Created interaction ratios")
        
        new_features_count = len(self.df.columns) - original_col_count
        print(f"\n Created {new_features_count} new features")
        
        # Show new features
        new_features = list(set(self.df.columns) - set(self.original_columns))
        print(f"\n New features created:")
        for i, feature in enumerate(sorted(new_features), 1):
            print(f"   {i:2}. {feature}")
    
    def save_to_processed_folder(self, filename="cleaned_tiktok_data.csv"):
        """
        Save cleaned data to Data/Processed folder
        """
        try:
            # Create Processed folder if it doesn't exist
            processed_folder = Path("Data/Processed")
            processed_folder.mkdir(parents=True, exist_ok=True)
            
            # Define file paths
            csv_path = processed_folder / filename
            excel_path = processed_folder / filename.replace('.csv', '.xlsx')
            
            # Save to CSV
            self.df.to_csv(csv_path, index=False)
            
            # Save to Excel
            self.df.to_excel(excel_path, index=False)
            
            # Create a summary file
            self.create_summary_file(processed_folder)
            
            print(f"\n Data saved to processed folder:")
            print(f"   CSV: {csv_path}")
            print(f"   Excel: {excel_path}")
            
            # Display file sizes
            import os
            csv_size = os.path.getsize(csv_path) / 1024  # KB
            excel_size = os.path.getsize(excel_path) / 1024  # KB
            
            print(f"   File sizes: CSV={csv_size:.1f}KB, Excel={excel_size:.1f}KB")
            
            return csv_path, excel_path
            
        except Exception as e:
            print(f" Error saving to processed folder: {e}")
            return None, None
    
    def create_summary_file(self, processed_folder):
        """
        Create a summary of the cleaning process
        """
        summary = {
            'cleaning_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'original_shape': list(self.original_shape),
            'final_shape': list(self.df.shape),
            'original_columns': self.original_columns,
            'new_columns': list(set(self.df.columns) - set(self.original_columns)),
            'missing_values_handled': self.df.isnull().sum().sum() == 0,
            'duplicates_removed': self.original_shape[0] - self.df.shape[0],
            'data_sample': {
                'first_5_rows': self.df.head().to_dict('records'),
                'last_5_rows': self.df.tail().to_dict('records')
            },
            'column_statistics': {
                col: {
                    'dtype': str(self.df[col].dtype),
                    'unique_values': int(self.df[col].nunique()) if self.df[col].dtype == 'object' or self.df[col].dtype.name == 'category' else None,
                    'null_count': int(self.df[col].isnull().sum()),
                    'min': float(self.df[col].min()) if pd.api.types.is_numeric_dtype(self.df[col]) else None,
                    'max': float(self.df[col].max()) if pd.api.types.is_numeric_dtype(self.df[col]) else None,
                    'mean': float(self.df[col].mean()) if pd.api.types.is_numeric_dtype(self.df[col]) else None,
                    'median': float(self.df[col].median()) if pd.api.types.is_numeric_dtype(self.df[col]) else None
                } for col in self.df.columns
            }
        }
        
        # Save summary as JSON
        import json
        summary_path = processed_folder / "cleaning_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        print(f"   📄 Summary: {summary_path}")
    
    def get_cleaning_report(self):
        """
        Generate a cleaning report
        """
        print("\n" + "="*60)
        print("CLEANING REPORT")
        print("="*60)
        
        report = {
            "Data Shape": f"{self.original_shape} → {self.df.shape}",
            "Rows Removed": self.original_shape[0] - self.df.shape[0],
            "Columns Added": len(self.df.columns) - len(self.original_columns),
            "Missing Values": "None" if self.df.isnull().sum().sum() == 0 else f"{self.df.isnull().sum().sum()} remaining",
            "Date Range": f"{self.df['Date_Posted'].min().date()} to {self.df['Date_Posted'].max().date()}" if 'Date_Posted' in self.df.columns else "N/A",
            "Reach Statistics": {
                "Min": f"{self.df['Reach'].min():,}",
                "Max": f"{self.df['Reach'].max():,}",
                "Mean": f"{self.df['Reach'].mean():,.0f}",
                "Median": f"{self.df['Reach'].median():,.0f}"
            } if 'Reach' in self.df.columns else "N/A",
            "Top 5 Features by Correlation with Reach": self.get_top_correlations()
        }
        
        for key, value in report.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")
        
        return report
    
    def get_top_correlations(self, n=5):
        """
        Get top correlations with Reach
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        if 'Reach' in numerical_cols and len(numerical_cols) > 1:
            correlations = self.df[numerical_cols].corr()['Reach'].sort_values(ascending=False)
            top_corr = {}
            for feature, corr in correlations.items():
                if feature != 'Reach' and not pd.isna(corr):
                    top_corr[feature] = round(corr, 3)
                    if len(top_corr) >= n:
                        break
            return top_corr
        return {}

if __name__ == "__main__":
    """
    Test the DataCleaner independently
    """
    print("Testing DataCleaner...")
    print("="*60)
    
    # First check if we can load data
    try:
        from load_data import DataLoader
        
        print("1. Loading data...")
        loader = DataLoader()
        df = loader.load_data()
        
        if df is not None:
            print("\n2. Cleaning data...")
            cleaner = DataCleaner(df)
            cleaned_df = cleaner.clean_data(
                save_to_processed=True,
                filename="tiktok_cleaned.csv"
            )
            
            print("\n3. Generating report...")
            cleaner.get_cleaning_report()
            
            print("\n" + "="*60)
            print("CLEANING COMPLETE!")
            print("="*60)
            print(f"\n Check Data/Processed/ for cleaned files")
            print(f" Original data: {df.shape}")
            print(f" Cleaned data: {cleaned_df.shape}")
            print(f" New features: {len(cleaned_df.columns) - len(df.columns)}")
            
            # Show sample of new data
            print(f"\n📋 Sample of cleaned data (first 3 rows):")
            print(cleaned_df[['Post_ID', 'Date_Posted', 'Reach', 'Engagement_Rate', 'Time_Category', 'Content_Type']].head(3).to_string(index=False))
            
        else:
            print(" Failed to load data for testing")
            
    except ImportError:
        print("Cannot import load_data module")
        print("Creating sample data for testing...")
        
        # Create sample data for testing
        np.random.seed(42)
        test_data = {
            'Post_ID': range(1, 101),
            'Date_Posted': pd.date_range('2024-01-01', periods=100),
            'Reach': np.random.randint(10000, 30000, 100),
            'Likes': np.random.randint(500, 1500, 100),
            'Comments': np.random.randint(10, 50, 100),
            'Shares': np.random.randint(20, 50, 100),
            'Duration_Seconds': np.random.randint(15, 120, 100),
            'Hashtags_Count': np.random.randint(1, 15, 100),
            'Caption_Length': np.random.randint(10, 80, 100),
            'Post_Hour': np.random.randint(0, 24, 100)
        }
        df = pd.DataFrame(test_data)
        
        print(f"Created test data with {len(df)} rows")
        
        print("\n2. Cleaning test data...")
        cleaner = DataCleaner(df)
        cleaned_df = cleaner.clean_data(
            save_to_processed=True,
            filename="test_cleaned.csv"
        )
        
        print("\n" + "="*60)
        print("TEST COMPLETE!")
        print("="*60)
        print(f"\n Check Data/Processed/ for test files")