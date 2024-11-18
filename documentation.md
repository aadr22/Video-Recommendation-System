# Video Recommendation System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Data Flow](#data-flow)
5. [Installation & Setup](#installation--setup)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Metrics & Monitoring](#metrics--monitoring)
9. [User Interface](#user-interface)

## System Overview

The Video Recommendation System is a comprehensive solution that provides personalized video recommendations to users based on their viewing history, engagement patterns, and content preferences. The system combines content-based filtering with engagement metrics to deliver relevant recommendations through both an API and a Streamlit-based web interface.

### Key Features
- Real-time data processing and updates
- Personalized recommendations based on user history
- Cold-start handling for new users
- Engagement score calculation
- Interactive visualization of recommendations
- Recommendation quality metrics (Precision@K, Recall@K)

## Architecture

The system consists of four main components:

1. **Data Processor (`data_processor.py`)**
   - Handles data ingestion from the SocialVerse API
   - Performs data preprocessing and cleaning
   - Updates data every 2 minutes
   - Generates engagement scores and post summaries

2. **Recommender Engine (`recommender.py`)**
   - Implements the core recommendation logic
   - Handles both personalized and cold-start recommendations
   - Calculates recommendation metrics
   - Manages content-based filtering using TF-IDF

3. **Web Interface (`app.py`)**
   - Provides a Streamlit-based user interface
   - Visualizes recommendations and metrics
   - Displays user preference analytics

4. **Main Application (`main.py`)**
   - Orchestrates the system components
   - Provides example usage implementations

## Components

### Data Processor
```python
class DataProcessor:
    - Fetches data from SocialVerse API
    - Preprocesses and cleans data
    - Calculates engagement scores
    - Generates post summaries
    - Schedules regular updates
```

Key Functions:
- `fetch_data(endpoint)`: Retrieves data from API endpoints
- `preprocess_data()`: Handles data cleaning and transformation
- `generate_detailed_post_summary(row)`: Creates comprehensive post summaries
- `convert_lists_to_strings(df)`: Handles list-type data conversion

### Recommender Engine
```python
class VideoRecommender:
    - Generates personalized recommendations
    - Handles cold-start scenarios
    - Manages content-based filtering
    - Calculates similarity scores
```

Key Functions:
- `get_recommendations(username)`: Generates personalized recommendations
- `_get_personalized_recommendations(username, user_history)`: Creates user-specific recommendations
- `_get_cold_start_recommendations()`: Handles new users
- `_prepare_content_features()`: Processes content for TF-IDF vectorization

### Metrics Calculator
```python
class RecommendationMetrics:
    - Calculates precision and recall metrics
    - Logs metric calculations
    - Provides performance insights
```

Key Functions:
- `calculate_precision_at_k(recommendations, actual_interactions, k)`: Calculates precision@k
- `calculate_recall_at_k(recommendations, actual_interactions, k)`: Calculates recall@k

## Data Flow

1. **Data Ingestion**
   ```
   SocialVerse API → Data Processor → Preprocessed CSV Files
   ```

2. **Recommendation Generation**
   ```
   User Request → Load User History → Generate Recommendations → Calculate Metrics → Return Results
   ```

3. **Visualization Flow**
   ```
   User Selection → Fetch Recommendations → Generate Visualizations → Display Results
   ```

## Installation & Setup

### Prerequisites
- Python 3.8+
- pandas
- numpy
- scikit-learn
- streamlit
- plotly
- requests
- schedule

### Installation Steps
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure API credentials in `data_processor.py`
4. Run the application:
   ```bash
   python main.py
   ```

## API Reference

### Data Processor API
```python
preprocess_data() → DataFrame
# Returns preprocessed data for recommendation engine

fetch_data(endpoint: str) → dict
# Fetches raw data from specified API endpoint
```

### Recommender API
```python
get_recommendations(username: str) → dict
# Returns recommendations for specified user
# Response format:
{
    "status": "success|error",
    "message": str,
    "recommendations": List[int]
}
```

### Metrics API
```python
calculate_precision_at_k(recommendations: list, actual_interactions: list, k: int) → float
calculate_recall_at_k(recommendations: list, actual_interactions: list, k: int) → float
```

## Configuration

### API Configuration
```python
BASE_URL = "https://api.socialverseapp.com"
HEADERS = {
    "Flic-Token": "your_token_here"
}
```

### Engagement Score Weights
```python
view_weight = 0.3
upvote_weight = 0.5
comment_weight = 0.2
```

## Metrics & Monitoring

### Logging
- Application logs: `recommender.log`
- Metrics logs: `metrics.log`
- Data processing logs: `data_processing.log`

### Performance Metrics
- Precision@K
- Recall@K
- Engagement scores
- Processing time
- API response times

## User Interface

The Streamlit interface provides:
1. User selection dropdown
2. Recommendation generation button
3. Results visualization:
   - Recommended videos table
   - Engagement score bar chart
   - Category distribution pie chart
4. Performance metrics display

### Usage Example
```python
app = StreamlitRecommenderApp()
app.run()
```

Access the interface at `http://localhost:8501` after running the application.
