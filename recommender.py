import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import logging
import traceback
import json
from datetime import datetime

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for handling NumPy data types.
    
    This encoder converts NumPy integers, floats, and arrays to their Python equivalents
    for proper JSON serialization.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, np.ndarray)):
            return obj.item() if isinstance(obj, (np.integer, np.floating)) else obj.tolist()
        return super().default(obj)

class RecommendationMetrics:
    """
    A class for calculating and evaluating recommendation system metrics.
    
    This class provides methods to calculate standard recommendation system metrics
    such as Precision@K and Recall@K, with built-in logging capabilities.
    """
    
    def __init__(self):
        """Initialize the RecommendationMetrics class with logging configuration."""
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging settings for the metrics calculation process."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('metrics.log'),
                logging.StreamHandler()
            ]
        )

    def calculate_precision_at_k(self, recommendations: list, actual_interactions: list, k: int) -> float:
        """
        Calculate Precision@K metric for recommendation evaluation.
        
        Args:
            recommendations (list): List of recommended item IDs
            actual_interactions (list): List of items the user actually interacted with
            k (int): Number of recommendations to consider
        
        Returns:
            float: Precision@K score as a percentage (0-100)
        """
        if not recommendations or not actual_interactions:
            logging.warning("Empty recommendations or actual interactions")
            return 0.0
        rec_set = set(map(str, recommendations[:k]))
        int_set = set(map(str, actual_interactions))
        precision_at_k = len(rec_set.intersection(int_set)) / k
        logging.info(f"Precision@{k} calculated: {precision_at_k:.2f}")
        return precision_at_k * 100

    def calculate_recall_at_k(self, recommendations: list, actual_interactions: list, k: int) -> float:
        """
        Calculate Recall@K metric for recommendation evaluation.
        
        Args:
            recommendations (list): List of recommended item IDs
            actual_interactions (list): List of items the user actually interacted with
            k (int): Number of recommendations to consider
        
        Returns:
            float: Recall@K score as a percentage (0-100)
        """
        if not recommendations or not actual_interactions:
            logging.warning("Empty recommendations or actual interactions")
            return 0.0
        rec_set = set(map(str, recommendations[:k]))
        int_set = set(map(str, actual_interactions))
        recall_at_k = len(rec_set.intersection(int_set)) / len(int_set) if len(int_set) > 0 else 0.0
        logging.info(f"Recall@{k} calculated: {recall_at_k:.2f}")
        return recall_at_k * 100

class VideoRecommender:
    """
    A video recommendation system that combines content-based and engagement-based filtering.
    
    This class implements a hybrid recommendation system that considers both video content
    and user engagement metrics to generate personalized video recommendations.
    """
    
    def __init__(self):
        """
        Initialize the VideoRecommender with necessary components and data.
        
        Sets up logging, metrics calculation, loads required data, and initializes
        the content-based filtering components.
        """
        self._setup_logging()
        self.metrics = RecommendationMetrics()
        self.load_data()
        self._initialize_components()

    def _setup_logging(self):
        """Configure logging settings for the recommendation system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('recommender.log'),
                logging.StreamHandler()
            ]
        )

    def load_data(self):
        """
        Load and validate necessary data from CSV files.
        
        Raises:
            Exception: If data loading fails or required files are missing
        """
        try:
            self.posts_df = pd.read_csv('data/preprocessed_data.csv')
            self.users_df = pd.read_csv('data/users_data.csv')
            self._validate_data()
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

    def _initialize_components(self):
        """Initialize TF-IDF vectorizer and scaler for content-based filtering."""
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.scaler = MinMaxScaler()
        self._prepare_content_features()

    def _prepare_content_features(self):
        """
        Prepare content features using TF-IDF vectorization on enriched post summaries.
        
        Combines post summaries with keywords and emotions for better content representation.
        """
        if 'post_summary' in self.posts_df.columns:
            self.posts_df['enriched_summary'] = self.posts_df.apply(
                lambda row: f"{row['post_summary']} {row.get('keywords', '')} {row.get('emotions', '')}", 
                axis=1
            )
            self.content_features = self.vectorizer.fit_transform(
                self.posts_df['enriched_summary'].fillna('')
            )

    def _validate_data(self) -> bool:
        """
        Validate the loaded data for required columns and data integrity.
        
        Returns:
            bool: True if validation passes
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = {'id', 'username', 'category_name', 'engagement_score'}
        
        if not required_columns.issubset(self.posts_df.columns):
            raise ValueError(f"Missing required columns: {required_columns - set(self.posts_df.columns)}")

        self.posts_df['engagement_score'] = self.posts_df['engagement_score'].fillna(0)
        self.posts_df.drop_duplicates(subset=['id'], keep='first', inplace=True)
        
        return True

    def _get_personalized_recommendations(self, username: str, user_history: pd.DataFrame) -> list:
        """
        Generate personalized recommendations based on user history.
        
        Args:
            username (str): Target user's username
            user_history (pd.DataFrame): User's past interactions
            
        Returns:
            list: List of recommended video IDs
        """
        try:
            # Get user's preferred categories
            category_counts = user_history['category_name'].value_counts()
            preferred_categories = category_counts[category_counts >= 2].index.tolist()

            if not preferred_categories:
                preferred_categories = user_history['category_name'].unique()

            # Filter by categories and engagement
            category_mask = (
                self.posts_df['category_name'].isin(preferred_categories) &
                (self.posts_df['engagement_score'] > 0)
            )
            category_recs = self.posts_df[category_mask].nlargest(3, 'engagement_score')['id'].tolist()

            # Get general recommendations
            general_mask = (
                ~self.posts_df['id'].isin(category_recs) &
                (self.posts_df['engagement_score'] > 0)
            )
            general_recs = self.posts_df[general_mask].nlargest(2, 'engagement_score')['id'].tolist()

            return category_recs + general_recs

        except Exception as e:
            logging.error(f"Error in personalized recommendations: {e}")
            return []

    def _get_cold_start_recommendations(self) -> list:
        """
        Generate recommendations for new users with no history.
        
        Returns:
            list: List of recommended video IDs based on overall popularity
        """
        try:
            top_videos = self.posts_df.nlargest(5, 'engagement_score')['id'].tolist()
            return top_videos

        except Exception as e:
            logging.error(f"Error in cold-start recommendations: {e}")
            return []

    def get_recommendations(self, username: str) -> dict:
        """
        Generate video recommendations for a given user.
        
        Args:
            username (str): Target user's username
            
        Returns:
            dict: Dictionary containing recommendation results with keys:
                - status: Success/error status
                - message: Status message
                - recommendations: List of recommended video IDs
        """
        try:
            if username not in self.users_df['username'].values:
                return {
                    "status": "error",
                    "message": "User not found",
                    "recommendations": []
                }

            user_history = self.posts_df[self.posts_df['username'] == username]
            recommendations = (
                self._get_personalized_recommendations(username, user_history)
                if not user_history.empty else
                self._get_cold_start_recommendations()
            )

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
