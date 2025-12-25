# ML-Group-Project---Spotify-playlist-reconstruction
In this project, we analyzed Spotify audio features (danceability, energy, tempo, valence) to predict user preferences and classify songs using multiple supervised learning models—Logistic Regression, KNN, SVM, and Decision Trees. We also designed a hybrid recommendation system that balances individual taste with shared family listening patterns.

---

📌 Project Context

After a simulated hacker attack on Spotify servers, only one **mixed playlist** per family account remained, containing songs from all users and years.  
Partial information about which user and which year each song belongs to was recovered, but some songs remain unlabeled.

The goal of this project is to:
- Reconstruct missing playlist information using machine learning
- Design a recommendation system suitable for Spotify family accounts

---
🎯 Project Objectives

 ```bash

- Explore and clean Spotify audio feature data
- Train supervised ML models to predict user preferences
- Design a hybrid recommendation system
- Evaluate recommendation quality using ranking metrics
- Apply real-world AI concepts taught in the course
```
---

🧠 Project Structure
 ```bash
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
```

---
🔍 Task 1 – Data Exploration & Supervised Learning

Steps Performed
 ```bash
- Data cleaning (missing values, duplicates)
- Feature selection
- Standardization using StandardScaler
- Train–test split (80/20)
- Model training and evaluation
```

Models Used
```bash
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
```
Evaluation Metrics
```bash
- Accuracy
- Precision
- Recall
- F1-score
```

---
🤖 Task 2 – Hybrid Recommendation System

1️⃣ Content-Based Filtering
```bash
Predicts how much a user will like a song based on its features.

Input:
Audio features + metadata

Model:
Scikit-learn classifiers (Logistic Regression, KNN, SVM)
```

2️⃣ Collaborative Component
```bash
- Captures shared preferences across users:
- Songs listened to by multiple users get higher relevance.
- Simulates collaborative filtering without matrix factorization.
```

3️⃣ Hybrid Scoring

Final recommendation score:
 ```bash

Hybrid Score = α × Content Score + (1 − α) × Collaborative Score
```

This allows:
```bash
- Cold-start handling
- Personalized recommendations
- Balanced exploration & exploitation
```

---
📈 Evaluation Strategy

Temporal train–test split (past → future)

Ranking-based evaluation:
```bash
- Precision@K
- Recall@K
- NDCG@K
- Model tuning via GridSearchCV
```
---
📊 Dataset Description

mixed_playlist.csv

⚠️ **Important:**  
The dataset is **not included in this repository** and must be **uploaded manually** before running the notebook.
 ```bash
### Dataset Description
- Total songs: 3,600
- Labeled songs: 3,500 (user + year known)
- Unlabeled songs: 100 (user and year missing)
- Features include audio characteristics and popularity-related metadata
```
---
📌 Key Learnings
 ```bash

- How ML supports personalized decision-making
- Trade-offs between content-based and collaborative filtering
- Importance of feature engineering in recommender systems
- Application of supervised ML to real-world business problems
```
---
🚀 Future Enhancements
 ```bash

- Matrix factorization (ALS / SVD)
- Deep learning recommender models
- Real-time recommendation API
- Dashboard visualization
```
---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ML-Group-Project---Spotify-playlist-reconstruction.git
Upload mixed_playlist.csv to the project directory

Open the Jupyter notebook:
```bash
 Spotify machine learning project  Task1 .ipynb
```
Run all cells from top to bottom
---
Tools & Technologies
 ```bash
Python
Jupyter Notebook
Pandas, NumPy
Scikit-learn
Machine Learning (Classification, Evaluation)
```
---
Disclaimer
This project is for academic purposes only and is not affiliated with or endorsed by Spotify.
