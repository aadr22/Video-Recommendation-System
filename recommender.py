import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import logging
import traceback
import json
from datetime import datetime

# Custom JSON encoder to handle numpy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.ndarray)):
            return obj.item() if isinstance(obj, (np.integer, np.floating)) else obj.tolist()
        return super().default(obj)

# Class to calculate recommendation metrics like CTR and MAP
class RecommendationMetrics:
    def __init__(self):
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('metrics.log'),
                logging.StreamHandler()
            ]
        )

    def calculate_precision_at_k(self, recommendations: list, actual_interactions: list, k: int) -> float:
        if not recommendations or not actual_interactions:
            logging.warning("Empty recommendations or actual interactions")
            return 0.0
        rec_set = set(map(str, recommendations[:k]))
        int_set = set(map(str, actual_interactions))
        precision_at_k = len(rec_set.intersection(int_set)) / k
        logging.info(f"Precision@{k} calculated: {precision_at_k:.2f}")
        return precision_at_k * 100

    def calculate_recall_at_k(self, recommendations: list, actual_interactions: list, k: int) -> float:
        if not recommendations or not actual_interactions:
            logging.warning("Empty recommendations or actual interactions")
            return 0.0
        rec_set = set(map(str, recommendations[:k]))
        int_set = set(map(str, actual_interactions))
        recall_at_k = len(rec_set.intersection(int_set)) / len(int_set) if len(int_set) > 0 else 0.0
        logging.info(f"Recall@{k} calculated: {recall_at_k:.2f}")
        return recall_at_k * 100

# Video recommender system class
class VideoRecommender:
    def __init__(self):
        self._setup_logging()
        self.metrics = RecommendationMetrics()  # Initialize RecommendationMetrics instance directly.
        self.load_data()
        self._initialize_components()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('recommender.log'),
                logging.StreamHandler()
            ]
        )

    # Load data from CSV files
    def load_data(self):
        try:
            self.posts_df = pd.read_csv('data/preprocessed_data.csv')
            self.users_df = pd.read_csv('data/users_data.csv')
            self._validate_data()
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

    # Initialize components for content-based filtering
    def _initialize_components(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.scaler = MinMaxScaler()
        self._prepare_content_features()

    # Prepare content features using TF-IDF vectorization on enriched post summaries
    def _prepare_content_features(self):
        if 'post_summary' in self.posts_df.columns:
            # Enrich post summary with additional features like keywords and emotions
            self.posts_df['enriched_summary'] = self.posts_df.apply(
                lambda row: f"{row['post_summary']} {row.get('keywords', '')} {row.get('emotions', '')}", axis=1)
            
            # Apply TF-IDF vectorization on the enriched summary
            self.content_features = self.vectorizer.fit_transform(self.posts_df['enriched_summary'].fillna(''))

    # Validate the data to ensure required columns are present
    def _validate_data(self) -> bool:
        required_columns = {'id', 'username', 'category_name', 'engagement_score'}
        
        if not required_columns.issubset(self.posts_df.columns):
            raise ValueError(f"Missing required columns: {required_columns - set(self.posts_df.columns)}")

        self.posts_df['engagement_score'] = self.posts_df['engagement_score'].fillna(0)
        self.posts_df.drop_duplicates(subset=['id'], keep='first', inplace=True)
        
        return True

    def _get_personalized_recommendations(self, username: str, user_history: pd.DataFrame) -> list:
        try:
            # Get user's preferred categories based on their interaction history
            category_counts = user_history['category_name'].value_counts()
            preferred_categories = category_counts[category_counts >= 2].index.tolist()

            # Fallback to all categories if no strong preference found
            if not preferred_categories:
                preferred_categories = user_history['category_name'].unique()

            # Filter posts by user's preferred categories
            category_mask = (
                self.posts_df['category_name'].isin(preferred_categories) &
                (self.posts_df['engagement_score'] > 0)
            )

            # Get top recommendations from preferred categories
            category_recs = self.posts_df[category_mask].nlargest(3, 'engagement_score')['id'].tolist()

            # General fallback recommendations (cold-start)
            general_mask = (
                ~self.posts_df['id'].isin(category_recs) &
                (self.posts_df['engagement_score'] > 0)
            )
            
            general_recs = self.posts_df[general_mask].nlargest(2, 'engagement_score')['id'].tolist()

            return category_recs + general_recs

        except Exception as e:
            logging.error(f"Error in personalized recommendations: {e}")
            return []

    # Fallback for cold start users with no history - recommend popular videos by engagement score
    def _get_cold_start_recommendations(self) -> list:
        try:
            top_videos = self.posts_df.nlargest(5, 'engagement_score')['id'].tolist()
            return top_videos

        except Exception as e:
            logging.error(f"Error in cold-start recommendations: {e}")
            return []

    def get_recommendations(self, username: str) -> dict:
        try:
            if username not in self.users_df['username'].values:
                return {
                    "status": "error",
                    "message": "User not found",
                    "recommendations": []
                }

            # Get user's interaction history
            user_history = self.posts_df[self.posts_df['username'] == username]

            # Get personalized or cold start recommendations based on history availability.
            recommendations = (
                self._get_personalized_recommendations(username, user_history)
                if not user_history.empty else
                self._get_cold_start_recommendations()
            )

            # Convert to integer IDs (no limit to top 5)
            recommendations = [int(rec) for rec in recommendations]

            return {
                "status": "success",
                "message": "Recommendations generated successfully",
                "recommendations": recommendations,
            }

        except Exception as e:
            logging.error(f"Error generating recommendations: {e}")
            return {
                "status": "error",
                "message": str(e),
                "recommendations": []
            }