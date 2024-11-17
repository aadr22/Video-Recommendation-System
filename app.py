import streamlit as st
import pandas as pd
from recommender import VideoRecommender, RecommendationMetrics
import plotly.express as px
import logging

class StreamlitRecommenderApp:
    def __init__(self):
        # Initialize the recommender and metrics classes
        self.recommender = VideoRecommender()
        self.metrics = RecommendationMetrics()  # Initialize RecommendationMetrics

    def load_user_data(self):
        # Load user data from CSV file
        return pd.read_csv('data/users_data.csv')

    def calculate_metrics(self, recommendations, user_history):
        try:
            # Filter user interactions based on view, upvote, or comment counts
            actual_interactions = user_history[
                (user_history['view_count'] >= 1) |
                (user_history['upvote_count'] >= 1) |
                (user_history['comment_count'] >= 1)
            ]['id'].tolist()

            if not actual_interactions:
                logging.warning("No meaningful user interactions found")
                return 0.0, 0.0

            # Calculate Precision@K and Recall@K for the recommendations
            precision_at_k = self.metrics.calculate_precision_at_k(recommendations, actual_interactions, k=len(recommendations))
            recall_at_k = self.metrics.calculate_recall_at_k(recommendations, actual_interactions, k=len(recommendations))
            return precision_at_k, recall_at_k

        except Exception as e:
            logging.error(f"Metrics calculation failed: {e}")
            return 0.0, 0.0

    def create_engagement_plot(self, recommendations_df):
        # Create a bar chart to visualize engagement scores of recommended videos
        fig = px.bar(
            recommendations_df,
            x='id',
            y='engagement_score',
            title='Engagement Scores for Recommended Videos'
        )
        return fig

    def run(self):
        st.title("Video Recommendation System")
        
        # Load user data and populate username selection dropdown
        users_df = self.load_user_data()
        available_users = users_df['username'].tolist()
        username = st.selectbox("Select a username", options=available_users)

        if st.button("Get Recommendations"):
            with st.spinner("Generating recommendations..."):
                # Generate recommendations for the selected user
                results = self.recommender.get_recommendations(username)

                if results["status"] == "success":
                    st.success("Recommendations generated successfully!")
                    st.subheader("Recommended Videos")

                    recommendations = results["recommendations"]
                    recommended_videos = self.recommender.posts_df[
                        self.recommender.posts_df['id'].isin(recommendations)
                    ]

                    # Display recommended videos and their details in a table
                    video_ids, categories, engagement_scores = [], [], []
                    for vid_id in recommendations:
                        video = recommended_videos[recommended_videos['id'] == vid_id].iloc[0]
                        video_ids.append(vid_id)
                        categories.append(video['category_name'])
                        engagement_scores.append(round(float(video['engagement_score']), 3))

                    metrics_df = pd.DataFrame({
                        'Video ID': video_ids,
                        'Category': categories,
                        'Engagement Score': engagement_scores
                    })
                    st.dataframe(metrics_df)

                    # Display engagement analysis plot
                    st.subheader("Engagement Analysis")
                    fig = self.create_engagement_plot(recommended_videos)
                    st.plotly_chart(fig)

                    # Display user's category preferences in a pie chart
                    st.subheader("Resonance Analysis")
                    user_history = self.recommender.posts_df[self.recommender.posts_df['username'] == username].copy()
                    
                    if not user_history.empty:
                        category_preferences = user_history['category_name'].value_counts()
                        fig_pie = px.pie(
                            values=category_preferences.values,
                            names=category_preferences.index,
                            title="User Category Distribution"
                        )
                        st.plotly_chart(fig_pie)
                    else:
                        st.info("New user - Recommendations based on general popularity")
                else:
                    st.error(results["message"])

if __name__ == "__main__":
    app = StreamlitRecommenderApp()
    app.run()