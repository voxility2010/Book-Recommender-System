# Book-Recommender-System
A hybrid recommendation engine built using the Goodbooks-10K
 dataset.
It combines Content-Based Filtering (TF-IDF + Cosine Similarity) and Collaborative Filtering (SVD) to deliver personalized, diverse, and explainable book suggestions.

🚀 Key Features

🔍 Content-Based Filtering:
Recommends books similar in title, author, and language code using TF-IDF and Cosine Similarity.

🤝 Collaborative Filtering:
Predicts ratings for unread books using Singular Value Decomposition (SVD) from the scikit-surprise library.

⚡ Hybrid Ensemble:
Combines both models using a weighted average:

Final Score
=
0.6
×
CF Score
+
0.4
×
Content Boost
Final Score=0.6×CF Score+0.4×Content Boost

This balances personalization and thematic similarity, improving both accuracy and diversity.

📈 Evaluation:
Implements Precision@10 to measure top-K recommendation performance.

📊 Dataset
File	Description	Size
books.csv	Metadata including titles, authors, language codes, and IDs	10,000 entries
ratings.csv	Explicit user ratings (1–5 stars)	~981,000 entries
🛠️ Setup & Installation
Prerequisites

Designed for Google Colab — leverages kagglehub for easy dataset download.

Installation Steps
# 1. Install numpy compatible with scikit-surprise
!pip install --quiet "numpy<2"

# 2. Core libraries
!pip install --quiet scikit-surprise pandas scikit-learn matplotlib

# 3. Kaggle data utility
!pip install --quiet kagglehub

⚙️ Running the Notebook

Upload the notebook → book_recommender_colab_AM343_AM337.ipynb to Google Colab
.

Run all cells sequentially — data will auto-download and preprocess.

View recommendations via the hybrid function.

🧠 Model Architecture
1️⃣ Content-Based Model

Method: TF-IDF + Cosine Similarity

Text Fields: title, authors, language_code

Function: content_recommender(book_id)

2️⃣ Collaborative Filtering Model

Method: SVD (Singular Value Decomposition)

Data Fields: user_id, book_id, rating

Function: cf_recommend(user_id)

3️⃣ Hybrid Model

Formula:

hybrid_score = 0.6 * cf_score + 0.4 * content_similarity


Function: hybrid_recommend(user_id)

Advantage:

Solves cold-start limitations of CF

Improves thematic relevance from content filtering

🧩 Evaluation Metric

Precision@K

def precision_at_k(recs, ground_truth, k=10):
    """
    Compute the proportion of top-K recommendations 
    that are relevant (rating >= 4).
    """


This metric evaluates how many of the top-K recommended books were actually liked by the user.

🚧 Future Enhancements
Goal	Description
🧮 Fix SVD Dependency	Resolve surprise import issues for stable model training
⚙️ Scalability	Replace full cosine matrix with Approximate Nearest Neighbors (FAISS / Annoy)
🎯 Hyperparameter Tuning	Optimize SVD parameters via cross-validation to minimize RMSE
🧠 Feature Augmentation	Add book descriptions, genres, or tags to enrich TF-IDF vectors
👀 Implicit Feedback	Explore ALS and implicit MF models for click/view-based data
📦 Tech Stack

Python 3.10+

Libraries:
pandas, scikit-learn, scikit-surprise, matplotlib, kagglehub

Environment: Google Colab

📈 Example Workflow
# Get top recommendations for a user
user_id = 123
recommendations = hybrid_recommend(user_id)
recommendations.head(10)

🧾 License

This project is open-sourced under the MIT License.
Feel free to fork, improve, and share!

💡 Acknowledgments

Dataset: Goodbooks-10K on Kaggle

Developed as part of the Machine Learning Applications Project (AM343 & AM337)
