# ML-Group-Project---Spotify-playlist-reconstruction
In this project, we analyzed Spotify audio features (danceability, energy, tempo, valence) to predict user preferences and classify songs using multiple supervised learning models—Logistic Regression, KNN, SVM, and Decision Trees. We also designed a hybrid recommendation system that balances individual taste with shared family listening patterns.


This project applies machine learning and data analytics to build a hybrid music recommendation system using Spotify listening data.
It combines content-based filtering and collaborative learning to support personalized decision-making in a multi-user (family) environment.

🎯 Project Objectives

Explore and clean Spotify audio feature data

Train supervised ML models to predict user preferences

Design a hybrid recommendation system

Evaluate recommendation quality using ranking metrics

Apply real-world AI concepts taught in the course

🧠 Project Structure
moneywiz-crm/
│
├── data/
│   └── mixed_playlist.csv
│
├── notebooks/
│   └── Spotify_machine_learning_project_Task1.ipynb
│
├── ppt/
│   └── Spotify_machine_learning_project.pptx
│
├── README.md
└── requirements.txt

📊 Dataset Description

The dataset includes Spotify track-level information:

Audio Features

danceability

energy

loudness

speechiness

acousticness

instrumentalness

liveness

valence

tempo

Metadata

popularity

year

user_id

song_id

Target Variable

liked → 1 if the user listened to the song, 0 otherwise

🔍 Task 1 – Data Exploration & Supervised Learning
Steps Performed

Data cleaning (missing values, duplicates)

Feature selection

Standardization using StandardScaler

Train–test split (80/20)

Model training and evaluation

Models Used

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Evaluation Metrics

Accuracy

Precision

Recall

F1-score

🤖 Task 2 – Hybrid Recommendation System
1️⃣ Content-Based Filtering

Predicts how much a user will like a song based on its features.

Input:
Audio features + metadata

Model:
Scikit-learn classifiers (Logistic Regression, KNN, SVM)

2️⃣ Collaborative Component

Captures shared preferences across users:

Songs listened to by multiple users get higher relevance.

Simulates collaborative filtering without matrix factorization.

3️⃣ Hybrid Scoring

Final recommendation score:

Hybrid Score = α × Content Score + (1 − α) × Collaborative Score


This allows:

Cold-start handling

Personalized recommendations

Balanced exploration & exploitation

📈 Evaluation Strategy

Temporal train–test split (past → future)

Ranking-based evaluation:

Precision@K

Recall@K

NDCG@K

Model tuning via GridSearchCV

🧪 Technologies Used

Python 3.x

Pandas & NumPy

Scikit-learn

Jupyter Notebook / VS Code

▶️ How to Run
Option 1: Jupyter Notebook
jupyter notebook


Open:

Spotify_machine_learning_project_Task1.ipynb

📌 Key Learnings

How ML supports personalized decision-making

Trade-offs between content-based and collaborative filtering

Importance of feature engineering in recommender systems

Application of supervised ML to real-world business problems

🚀 Future Enhancements

Matrix factorization (ALS / SVD)

Deep learning recommender models

Real-time recommendation API

Dashboard visualization (Streamlit)
