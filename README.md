# 🚀 ShopSmart AI: High-Performance E-commerce Recommendation Engine

ShopSmart AI is a production-ready, end-to-end recommendation system built to handle large-scale e-commerce data. It leverages **Collaborative Filtering** with the **ALS (Alternating Least Squares)** algorithm to provide personalized product suggestions from a dataset of over **42 million interactions**.

## 🏗️ Architecture Overview
The project is architected using a microservices-inspired approach, fully containerized with **Docker**:
- **Machine Learning Engine:** Built with the `Implicit` library for scalable matrix factorization.
- **Backend API:** High-performance inference layer developed with **FastAPI**.
- **Frontend UI:** Interactive dashboard built with **Streamlit** for real-time simulations.
- **Containerization:** Orchestrated via **Docker Compose** for seamless deployment.

## 📊 Dataset Insights
The model is trained on the **eCommerce Behavior Data from Multi Category Store** (Kaggle).
- **Size:** 42M+ rows (October 2019).
- **Events:** `view`, `cart`, and `purchase` interactions.
- **Complexity:** Handled extreme data sparsity and high cardinality (thousands of products/categories).

## ✨ Key Features
- **Personalized Recommendations:** Latent factor analysis for registered users.
- **Similar Product Discovery:** Item-to-item similarity using vector space distance.
- **Real-Time Cold Start Solution:** Dynamic session-based recommendations for new users based on instant cart actions.
- **Hybrid Catalog Generation:** Smart sampling of products across 50+ categories to prevent popularity bias.

## 🛠️ Tech Stack
- **Languages:** Python (Pandas, NumPy, Scipy)
- **ML:** Implicit (ALS), Scikit-Learn
- **API:** FastAPI, Uvicorn
- **UI:** Streamlit
- **DevOps:** Docker, Docker Compose

## 🚀 Getting Started

### Prerequisites
- Docker & Docker Compose installed on your machine.
- Trained model files (`.pkl`) placed in the `/models` directory.

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/iberkguven/ecommerce-recsys.git](https://github.com/iberkguven/ecommerce-recsys.git)
   cd ecommerce-recsys
