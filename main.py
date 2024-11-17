# main.py
from data_processor import preprocess_data
from recommender import VideoRecommender
import logging
import json

def main():
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # First, process the data
        processed_data = preprocess_data()
        if processed_data is None:
            logging.error("Data preprocessing failed")
            return

        # Initialize recommender
        recommender = VideoRecommender()

        # Example usage for existing user
        username = "kinha"
        result = recommender.get_top_5_recommendations(username)
        print(f"\nRecommendations for {username}:")
        print(json.dumps(result, indent=2))

        # Example usage for new user
        new_user = "newuser"
        result = recommender.get_top_5_recommendations(new_user)
        print(f"\nRecommendations for {new_user}:")
        print(json.dumps(result, indent=2))

    except Exception as e:
        logging.error(f"System error: {e}")
        return

if __name__ == "__main__":
    main()
