
# Video Recommendation System

This project is a video recommendation system that provides personalized video suggestions based on a user's interaction history (views, likes, comments). It uses content-based filtering techniques with TF-IDF vectorization and calculates recommendation metrics such as Precision@K and Recall@K.

---

## Features

- **Personalized Recommendations**: Users receive tailored video suggestions based on their past interactions (categories they prefer).
- **Cold Start Recommendations**: For new users with no history, popular videos are recommended.
- **Recommendation Metrics**: Precision@K and Recall@K are calculated to evaluate recommendation performance.
- **Data Processing**: The system fetches real-time data from an API and processes it every two minutes.

---

## Components

### 1. `app.py`
This file contains the Streamlit-based web interface where users can:
- Select their username.
- Get personalized video recommendations.
- View engagement analysis plots and resonance analysis (category preferences).

### 2. `data_processor.py`
This script handles:
- Fetching data from the API endpoints (`posts`, `users`, etc.).
- Preprocessing data (flattening nested fields).
- Calculating engagement scores.
- Generating post summaries.
- Saving the processed data into CSV files.

### 3. `recommender.py`
This module implements the recommendation logic by:
- Using TF-IDF vectorization to process post summaries.
- Calculating content similarity between posts.
- Generating recommendations based on the user's interaction history.

### 4. `main.py`
The main entry point of the system that:
- Preprocesses the data using `data_processor.py`.
- Generates example recommendations by calling methods from `recommender.py`.

---

## Installation

Follow the steps below to set up and run the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/video-recommendation-system.git
   cd video-recommendation-system
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Open the Streamlit app in your browser after running it locally.
2. Select a username from the dropdown menu.
3. Click **"Get Recommendations"** to see:
   - Personalized video suggestions.
   - Engagement analysis plots.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contribution

Contributions are welcome! Please fork this repository and submit a pull request for any improvements or feature additions.
