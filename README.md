# Book-Recommender-System
This project develops a hybrid book recommendation system using the Goodbooks-10k dataset. It integrates two powerful recommendation paradigms—Content-Based Filtering (TF-IDF) and Collaborative Filtering (SVD)—to provide personalized and diverse book suggestions.

🚀 Key Features
Content-Based Filtering: Recommends books based on textual similarity (Title, Author, Language Code) using TF-IDF and Cosine Similarity.

Collaborative Filtering: Predicts user ratings for unread books based on past rating patterns using the SVD (Singular Value Decomposition) algorithm.

Hybrid Ensemble: Combines the normalized SVD prediction score and a content-based boost factor to generate the final, weighted recommendations (Weighted Average with α=0.6).

Evaluation: Includes a framework for evaluating top-K recommendation performance using Precision@10.

📊 Dataset
The project uses the publicly available Goodbooks-10k dataset.

Data Component	Description	Size
books.csv	Book metadata (titles, authors, ratings, IDs)	10,000 entries
ratings.csv	Explicit user ratings (1-5 stars)	~981,000 entries

Export to Sheets

🛠️ Setup and Installation
Prerequisites
The code is designed to run primarily in a Google Colab environment, leveraging kagglehub for easy data access.

Environment Setup
The following commands install the required Python libraries. Note the necessity of installing a compatible version of numpy for the scikit-surprise library.

Bash

# 1. Install numpy version compatible with scikit-surprise
!pip install --quiet "numpy<2"

# 2. Install core libraries (scikit-surprise for CF, scikit-learn for TF-IDF)
!pip install --quiet scikit-surprise pandas scikit-learn matplotlib

# 3. Data download utility
!pip install --quiet kagglehub
Running the Notebook
Upload the book_recommender_colab_AM343_AM337.ipynb file to Google Colab.

Run all cells sequentially.

🧠 Model Details
Content-Based Model
The content_recommender function finds books with similar metadata to a given book_id.

Technique: TF-IDF with Cosine Similarity.

Text Source: title, authors, language_code fields.

Collaborative Filtering Model
The cf_recommend function predicts ratings for unrated items for a given user_id.

Technique: SVD (Singular Value Decomposition).

Data Used: user_id, book_id, rating.

Hybrid Model
The hybrid_recommend function combines the strengths of both models.

CF Score (α=0.6): The normalized SVD predicted rating for the item.

Content Boost (1−α=0.4): The maximum cosine similarity between the candidate item and the user's top 3 highest-rated items.

Benefit: Addresses the "cold start" problem better than pure CF and provides better topic alignment than pure CF.

📉 Evaluation
The recommendation quality is assessed using a Precision metric suited for top-K lists.

Python

def precision_at_k(recs, ground_truth, k=10):
    # Relevance is defined as a rating >= 4 stars.
    # ... implementation ...
⚙️ Future Enhancements
The following steps are recommended to optimize and productionize the recommender system:

Resolve SVD Dependency: The core Collaborative Filtering model requires the surprise module import error to be fixed for accurate training and evaluation.

Scalability: The current use of a full Cosine Similarity Matrix is memory-intensive. Transition the Content-Based lookups to use Approximate Nearest Neighbors (ANN) libraries (e.g., FAISS or Annoy) for efficient, large-scale search.

Hyperparameter Tuning: Use cross-validation on the SVD model to optimize parameters (e.g., n_factors, learning rate) to minimize the prediction RMSE.

Feature Augmentation: Incorporate rich textual features like book descriptions or tags into the TF-IDF vectorizer to enhance content representation.

Implicit Feedback: Explore dedicated implicit feedback models (like Alternating Least Squares) which are optimized for implicit user interactions (e.g., clicks, reads, views) common in production systems.
