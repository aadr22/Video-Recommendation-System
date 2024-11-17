import traceback
import requests
import pandas as pd
import logging
import os
import time
import json
import schedule
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)

# Define the API endpoint and headers
BASE_URL = "https://api.socialverseapp.com"
HEADERS = {
    "Flic-Token": "flic_b9c73e760ec8eae0b7468e7916e8a50a8a60ea7e862c32be44927f5a5ca69867"
}

# Fetch data from API
def fetch_data(endpoint):
    try:
        response = requests.get(endpoint, headers=HEADERS)
        response.raise_for_status()
        json_response = response.json()
        if 'posts' not in json_response and 'users' not in json_response:
            logging.error(f"'posts' or 'users' key not found in response: {json_response}")
            return None
        return json_response.get('posts') or json_response.get('users')
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from {endpoint}: {e}")
        return None

# Convert list columns to strings for hashability
def convert_lists_to_strings(df):
    for column in df.columns:
        if df[column].apply(lambda x: isinstance(x, list)).any():
            df[column] = df[column].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    return df

# Log errors in function execution
def log_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            traceback.print_exc()
            return None
    return wrapper

# Generate detailed post summary for each row of data
def generate_detailed_post_summary(row):
    components = []
    if 'key_events' in row and row['key_events']:
        components.append(f"Key Events: {row['key_events']}")
    if 'emotions' in row and row['emotions']:
        components.append(f"Emotions: {row['emotions']}")
    if 'concepts' in row and row['concepts']:
        components.append(f"Concepts: {row['concepts']}")
    if 'keywords' in row and row['keywords']:
        components.append(f"Keywords: {row['keywords']}")
    if 'topics_of_video' in row and row['topics_of_video']:
        components.append(f"Topics: {row['topics_of_video']}")
    if 'psychological_view' in row and row['psychological_view']:
        components.append(f"Psychological View: {row['psychological_view']}")
    if 'engagement_score' in row:
        components.append(f"Engagement Score: {row['engagement_score']}")
    
    return "; ".join(components) if components else "No summary available"

# Preprocess data function with error handling and scheduling updates every 2 minutes.
@log_errors
def preprocess_data():
    # Fetch data from API
    viewed_posts = fetch_data(f"{BASE_URL}/posts/view?page=1&page_size=1000")
    liked_posts = fetch_data(f"{BASE_URL}/posts/like?page=1&page_size=5")
    user_ratings = fetch_data(f"{BASE_URL}/posts/rating?page=1&page_size=5")
    post_summaries = fetch_data(f"{BASE_URL}/posts/summary/get?page=1&page_size=1000")
    all_users = fetch_data(f"{BASE_URL}/users/get_all?page=1&page_size=1000")

    if (viewed_posts is None or liked_posts is None or user_ratings is None or 
        post_summaries is None or all_users is None):
        logging.error("Failed to fetch one or more datasets. Exiting preprocessing.")
        return None

    try:
        # Convert to DataFrames
        viewed_df = pd.DataFrame(viewed_posts)
        liked_df = pd.DataFrame(liked_posts)
        ratings_df = pd.DataFrame(user_ratings)
        summaries_df = pd.DataFrame(post_summaries)
        users_df = pd.DataFrame(all_users)

        # Convert list columns to strings for hashability
        viewed_df = convert_lists_to_strings(viewed_df)
        liked_df = convert_lists_to_strings(liked_df)
        ratings_df = convert_lists_to_strings(ratings_df)
        summaries_df = convert_lists_to_strings(summaries_df)
        users_df = convert_lists_to_strings(users_df)

    except Exception as e:
        logging.error(f"Error converting data to DataFrame: {e}")
        return None

    # Flatten nested fields (if any exist)
    for df in [viewed_df, liked_df, ratings_df, summaries_df]:
        try:
            if 'category' in df.columns:
                df['category_id'] = df['category'].apply(lambda x: x['id'] if isinstance(x, dict) else None)
                df['category_name'] = df['category'].apply(lambda x: x['name'] if isinstance(x, dict) else None)
            if 'baseToken' in df.columns:
                df['baseToken_address'] = df['baseToken'].apply(lambda x: x['address'] if isinstance(x, dict) else None)
                df['baseToken_name'] = df['baseToken'].apply(lambda x: x['name'] if isinstance(x, dict) else None)
                df['baseToken_symbol'] = df['baseToken'].apply(lambda x: x['symbol'] if isinstance(x, dict) else None)
            # Drop nested columns after flattening them
            df.drop(columns=['category', 'baseToken'], inplace=True, errors='ignore')
        
        except Exception as e:
            logging.error(f"Error flattening nested fields: {e}")

    # Fill missing values with defaults
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

    for df in [viewed_df, liked_df, ratings_df, summaries_df]:
        for col, value in fill_values.items():
            if col in df.columns and df[col].isnull().any():
                df[col].fillna(value, inplace=True)

    # Remove duplicates based on certain columns like `id` or `title`
    try:
        for df in [viewed_df, liked_df, ratings_df, summaries_df]:
            id_columns = [col for col in ['id', 'title'] if col in df.columns]
            if id_columns:
                df.drop_duplicates(subset=id_columns, inplace=True)

    except Exception as e:
        logging.error(f"Error removing duplicates: {e}")
    
    # Combine DataFrames into one unified DataFrame
    try:
        combined_df = pd.concat([viewed_df, liked_df, ratings_df, summaries_df], ignore_index=True)

        # Remove duplicates from combined DataFrame based on `id`
        if 'id' in combined_df.columns:
            combined_df.drop_duplicates(subset=['id'], inplace=True)

    except Exception as e:
        logging.error(f"Error combining DataFrames: {e}")
    
    # Calculate engagement score (if required columns are present)
    required_columns = ['view_count', 'upvote_count', 'comment_count']
    
    try:
      if all(col in combined_df.columns for col in required_columns):
          combined_df['engagement_score'] = (
              combined_df['view_count'] * 0.3 +
              combined_df['upvote_count'] * 0.5 +
              combined_df['comment_count'] * 0.2 
          )
      else:
          logging.warning("Engagement score could not be calculated due to missing columns.")
          combined_df['engagement_score'] = 0  # Default value
    
      # Generate post summaries using helper function
      combined_df['post_summary'] = combined_df.apply(
          lambda row: generate_detailed_post_summary(row),
          axis=1
      )

      # Save processed data to CSV files
      os.makedirs('data', exist_ok=True)
      combined_df.to_csv('data/preprocessed_data.csv', index=False)
      users_df.to_csv('data/users_data.csv', index=False)

      logging.info("Data saved successfully to CSV files.")
    
      return combined_df

    except Exception as e:
      logging.error(f"Error during final processing steps: {e}")
      return None

# Main function with real-time processing every 2 minutes.
@log_errors
def main():
    
    def update_data():
      start_time = time.time()
      final_data = preprocess_data()
      
      if final_data is not None:
          logging.info("\n--- Data Update Summary ---")
          logging.info(f"Total rows: {len(final_data)}")
          logging.info("\nColumn Types:")
          logging.info(final_data.dtypes)
          end_time = time.time()
          logging.info(f"Data update completed in {end_time - start_time:.2f} seconds")
      else:
          logging.error("Data update failed")

    # Schedule updates every 2 minutes.
    schedule.every(2).minutes.do(update_data)

    update_data()  # Run initial processing
    
    # Launch Streamlit app after first run of data processing.
    os.system("streamlit run app.py")

    while True:
      schedule.run_pending()
      time.sleep(1)

if __name__ == "__main__":
   main()
