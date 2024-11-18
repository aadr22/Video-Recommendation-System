import traceback
import requests
import pandas as pd
import logging
import os
import time
import json
import schedule
from datetime import datetime

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)

# API Configuration
BASE_URL = "https://api.socialverseapp.com"
HEADERS = {
    "Flic-Token": "flic_b9c73e760ec8eae0b7468e7916e8a50a8a60ea7e862c32be44927f5a5ca69867"
}

def fetch_data(endpoint: str) -> dict | None:
    """
    Fetch data from the specified API endpoint with error handling.

    Args:
        endpoint (str): The full API endpoint URL to fetch data from

    Returns:
        dict | None: JSON response containing 'posts' or 'users' data, or None if request fails

    Raises:
        requests.exceptions.RequestException: If the API request fails
    """
    try:
        response = requests.get(endpoint, headers=HEADERS)
        response.raise_for_status()
        json_response = response.json()
        
        # Validate response structure
        if 'posts' not in json_response and 'users' not in json_response:
            logging.error(f"Invalid response structure. Expected 'posts' or 'users' key: {json_response}")
            return None
            
        return json_response.get('posts') or json_response.get('users')
        
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for {endpoint}: {e}")
        return None

def convert_lists_to_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert list-type columns to JSON strings for DataFrame compatibility.

    Args:
        df (pd.DataFrame): Input DataFrame with potential list-type columns

    Returns:
        pd.DataFrame: DataFrame with list columns converted to strings
    """
    for column in df.columns:
        if df[column].apply(lambda x: isinstance(x, list)).any():
            df[column] = df[column].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    return df

def log_errors(func):
    """
    Decorator for comprehensive error logging and handling.
    Catches and logs all exceptions that occur in the decorated function.

    Args:
        func: The function to be decorated

    Returns:
        wrapper: The wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            traceback.print_exc()
            return None
    return wrapper

def generate_detailed_post_summary(row: pd.Series) -> str:
    """
    Generate a comprehensive summary of a post from its attributes.

    Args:
        row (pd.Series): A row from the posts DataFrame containing post attributes

    Returns:
        str: Formatted string containing relevant post information and metrics
    """
    components = []
    
    # Map of field names to their display labels
    summary_fields = {
        'key_events': 'Key Events',
        'emotions': 'Emotions',
        'concepts': 'Concepts',
        'keywords': 'Keywords',
        'topics_of_video': 'Topics',
        'psychological_view': 'Psychological View',
        'engagement_score': 'Engagement Score'
    }
    
    # Build summary components
    for field, label in summary_fields.items():
        if field in row and row[field]:
            components.append(f"{label}: {row[field]}")
    
    return "; ".join(components) if components else "No summary available"

@log_errors
def preprocess_data() -> pd.DataFrame | None:
    """
    Main data processing pipeline that fetches, transforms, and analyzes social media data.
    
    The function:
    1. Fetches data from multiple API endpoints
    2. Converts responses to DataFrames
    3. Flattens nested JSON structures
    4. Handles missing values and duplicates
    5. Calculates engagement metrics
    6. Generates post summaries
    7. Saves processed data to CSV files

    Returns:
        pd.DataFrame | None: Combined and processed DataFrame, or None if processing fails
    """
    # Define endpoints for data fetching
    endpoints = {
        'viewed_posts': f"{BASE_URL}/posts/view?page=1&page_size=1000",
        'liked_posts': f"{BASE_URL}/posts/like?page=1&page_size=5",
        'user_ratings': f"{BASE_URL}/posts/rating?page=1&page_size=5",
        'post_summaries': f"{BASE_URL}/posts/summary/get?page=1&page_size=1000",
        'all_users': f"{BASE_URL}/users/get_all?page=1&page_size=1000"
    }
    
    # Fetch all datasets
    datasets = {name: fetch_data(endpoint) for name, endpoint in endpoints.items()}
    
    # Validate all required data was fetched
    if any(data is None for data in datasets.values()):
        logging.error("Failed to fetch one or more datasets. Aborting preprocessing.")
        return None

    try:
        # Convert to DataFrames
        dataframes = {
            name: convert_lists_to_strings(pd.DataFrame(data))
            for name, data in datasets.items()
        }

        # Flatten nested fields for post-related DataFrames
        post_dfs = ['viewed_posts', 'liked_posts', 'user_ratings', 'post_summaries']
        for df_name in post_dfs:
            df = dataframes[df_name]
            
            # Flatten category information
            if 'category' in df.columns:
                df['category_id'] = df['category'].apply(lambda x: x['id'] if isinstance(x, dict) else None)
                df['category_name'] = df['category'].apply(lambda x: x['name'] if isinstance(x, dict) else None)
            
            # Flatten token information
            if 'baseToken' in df.columns:
                df['baseToken_address'] = df['baseToken'].apply(lambda x: x['address'] if isinstance(x, dict) else None)
                df['baseToken_name'] = df['baseToken'].apply(lambda x: x['name'] if isinstance(x, dict) else None)
                df['baseToken_symbol'] = df['baseToken'].apply(lambda x: x['symbol'] if isinstance(x, dict) else None)
            
            # Remove original nested columns
            df.drop(columns=['category', 'baseToken'], inplace=True, errors='ignore')

        # Define default values for missing data
        fill_values = {
            'title': '',
            'category_id': None,
            'category_name': '',
            'baseToken_address': '',
            'baseToken_name': '',
            'baseToken_symbol': '',
            'upvote_count': 0,
            'view_count': 0,
            'exit_count': 0,
            'rating_count': 0,
            'average_rating': 0,
            'share_count': 0,
            'comment_count': 0,
        }

        # Fill missing values and remove duplicates
        for df_name in post_dfs:
            df = dataframes[df_name]
            
            # Fill missing values
            for col, value in fill_values.items():
                if col in df.columns and df[col].isnull().any():
                    df[col].fillna(value, inplace=True)
            
            # Remove duplicates
            id_columns = [col for col in ['id', 'title'] if col in df.columns]
            if id_columns:
                df.drop_duplicates(subset=id_columns, inplace=True)

        # Combine post-related DataFrames
        post_data = pd.concat([dataframes[name] for name in post_dfs], ignore_index=True)
        post_data.drop_duplicates(subset=['id'], inplace=True)

        # Calculate engagement score
        engagement_columns = ['view_count', 'upvote_count', 'comment_count']
        if all(col in post_data.columns for col in engagement_columns):
            post_data['engagement_score'] = (
                post_data['view_count'] * 0.3 +
                post_data['upvote_count'] * 0.5 +
                post_data['comment_count'] * 0.2 
            )
        else:
            logging.warning("Missing columns for engagement score calculation")
            post_data['engagement_score'] = 0

        # Generate post summaries
        post_data['post_summary'] = post_data.apply(generate_detailed_post_summary, axis=1)

        # Save processed data
        os.makedirs('data', exist_ok=True)
        post_data.to_csv('data/preprocessed_data.csv', index=False)
        dataframes['all_users'].to_csv('data/users_data.csv', index=False)

        logging.info(f"Successfully processed and saved data with {len(post_data)} posts")
        return post_data

    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        return None

@log_errors
def main():
    """
    Main execution function that:
    1. Initiates data processing
    2. Schedules regular updates
    3. Launches the Streamlit dashboard
    """
    def update_data():
        """
        Performs a single data update cycle and logs the results.
        """
        start_time = time.time()
        final_data = preprocess_data()
        
        if final_data is not None:
            processing_time = time.time() - start_time
            logging.info("\n=== Data Update Summary ===")
            logging.info(f"Total posts processed: {len(final_data)}")
            logging.info("\nColumn Types:")
            logging.info(final_data.dtypes)
            logging.info(f"Processing time: {processing_time:.2f} seconds")
        else:
            logging.error("Data update failed")

    # Schedule regular updates
    schedule.every(2).minutes.do(update_data)
    
    # Initial processing
    update_data()
    
    # Launch Streamlit dashboard
    os.system("streamlit run app.py")
    
    # Main update loop
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
