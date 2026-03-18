import os
import pickle
import logging
from src.data_pipeline import DataPipeline
from src.recommender import ALSRecommender

# Configuration
RAW_DATA_PATH = "data/raw/2019-Oct.csv"
MODEL_PATH = "models/als_model_best.pkl"
LOOKUP_PATH = "models/product_lookup.pkl"

def main():
    # Logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Ensure directory structure
    os.makedirs("models", exist_ok=True)

    logging.info("Starting training pipeline execution.")

    # Step 1: Data Processing
    pipeline = DataPipeline(file_path=RAW_DATA_PATH)
    
    logging.info("Loading dataset from: %s", RAW_DATA_PATH)
    df = pipeline.load_data()
    
    # Create Product Lookup for API Enrichment
    logging.info("Generating product lookup table.")
    product_lookup = pipeline.create_product_lookup(df)
    with open(LOOKUP_PATH, "wb") as f:
        pickle.dump(product_lookup, f)
    
    # Build Interaction Matrix
    logging.info("Building sparse interaction matrix.")
    sparse_matrix = pipeline.process_and_build_matrix(df)
    
    user_cats = pipeline.user_categories
    product_cats = pipeline.product_categories

    logging.info("Data processing complete. Matrix shape: %s Users x %s Products", 
                 sparse_matrix.shape[0], sparse_matrix.shape[1])

    # Step 2: Model Training & Serialization
    logging.info("Initializing ALS model training.")
    recommender = ALSRecommender()
    
    recommender.train_and_save(
        sparse_matrix=sparse_matrix,
        user_cats=user_cats,
        product_cats=product_cats,
        model_path=MODEL_PATH
    )

    logging.info("Training successful. Artifacts saved to 'models/' directory.")

if __name__ == "__main__":
    main()