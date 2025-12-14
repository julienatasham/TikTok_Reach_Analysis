import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    def __init__(self, data_path="Data/Raw/TikTok_Reach_Analysis_data.xlsx"):
        """
        Initialize DataLoader with path to data file
        """
        self.data_path = Path(data_path)
        
    def load_data(self, sheet_name="TikTok_Reach_Data"):
        """
        Load TikTok data from Excel file
        """
        try:
            df = pd.read_excel(self.data_path, sheet_name=sheet_name)
            print(f"✅ Data loaded successfully! Shape: {df.shape}")
            print(f"📊 Columns: {list(df.columns)}")
            print(f"📅 Date range: {df['Date_Posted'].min()} to {df['Date_Posted'].max()}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def get_data_info(self, df):
        """
        Get basic information about the dataset
        """
        print("=" * 50)
        print("DATASET INFORMATION")
        print("=" * 50)
        
        print(f"\n📈 Shape: {df.shape}")
        print(f"📊 Columns: {len(df.columns)}")
        
        print("\n📋 Column Information:")
        print(df.dtypes)
        
        print("\n🔍 Missing Values:")
        missing = df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")
        
        print("\n📊 Basic Statistics:")
        print(df.describe())
        
        return df.info()
    
    def save_processed_data(self, df, filename="processed_data.csv"):
        """
        Save processed data to CSV
        """
        output_path = Path("Data/Processed") / filename
        df.to_csv(output_path, index=False)
        print(f"✅ Processed data saved to: {output_path}")

if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    df = loader.load_data()
    if df is not None:
        loader.get_data_info(df)