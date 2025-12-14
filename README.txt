# TikTok Reach Analysis Project

## Project Overview
This project analyzes TikTok post performance data to understand factors affecting reach and engagement.

## Data Description
The dataset contains 500 TikTok posts with the following features:
- Post_ID: Unique identifier for each post
- Date_Posted: Date when the post was published
- Reach: Number of users who saw the post (target variable)
- Likes: Number of likes
- Comments: Number of comments
- Shares: Number of shares
- Duration_Seconds: Video length in seconds
- Hashtags_Count: Number of hashtags used
- Caption_Length: Length of caption in characters
- Post_Hour: Hour of day when posted (0-23)

## Project Structure
TIKTOK_REACH_ANALYSIS/
├── Data/
│ ├── Processed/
│ └── Raw/
├── TikTok_Reach_Analysis_outputs/
│ ├── figures/
│ └── saved_models/
├── scripts/
│ ├── init.py
│ ├── load_data.py
│ ├── clean_data.py
│ ├── eda.py
│ └── utils.py
├── models/
│ ├── init.py
│ ├── regression_model.py
│ ├── classification_model.py
│ └── time_series_model.py
├── main.txt
├── README.txt
└── requirements.txt

## Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place your data in `Data/Raw/`

## Usage
Run the main analysis: `python main.py`