import implicit
import pickle
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from scipy.sparse import csr_matrix

# Configuration of logging to track model inference and errors
logger = logging.getLogger(__name__)

class ALSRecommender:
    """
    ALSRecommender handles the lifecycle of the Matrix Factorization model,
    including training, serialization, and multi-strategy recommendation logic.
    """

    def __init__(self):
        """Initializes placeholders for model components and mapping categories."""
        self.model: Optional[implicit.als.AlternatingLeastSquares] = None
        self.user_categories: Optional[Any] = None
        self.product_categories: Optional[Any] = None
        self.sparse_matrix: Optional[csr_matrix] = None

    def train_and_save(self, 
                       sparse_matrix: csr_matrix, 
                       user_cats: Any, 
                       product_cats: Any, 
                       model_path: str = 'models/als_model_best.pkl') -> None:
        """
        Trains the Alternating Least Squares (ALS) model and persists 
        all necessary artifacts to a pickle file.
        """
        try:
            # Hyperparameters optimized for latent feature extraction
            self.model = implicit.als.AlternatingLeastSquares(
                factors=200, 
                regularization=0.2, 
                iterations=15, 
                random_state=42
            )
            
            self.model.fit(sparse_matrix, show_progress=True)
            
            # Encapsulate all components required for inference
            model_artifacts = {
                'model': self.model, 
                'user_categories': user_cats,
                'product_categories': product_cats,
                'sparse_user_item': sparse_matrix
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_artifacts, f)
                
            logger.info("Model training complete. Artifacts saved to %s", model_path)
            
        except Exception as e:
            logger.error("Critical failure during model training: %s", str(e))
            raise

    def load_model(self, model_path: str = 'models/als_model_best.pkl') -> None:
        """
        Loads the serialized model and index mappings into memory 
        to serve real-time requests.
        """
        try:
            with open(model_path, 'rb') as f:
                artifacts = pickle.load(f)
                
            self.model = artifacts['model']
            self.user_categories = artifacts['user_categories']
            self.product_categories = artifacts['product_categories']
            self.sparse_matrix = artifacts['sparse_user_item']
            
            logger.info("Inference engine ready. Model loaded from %s", model_path)
        except Exception as e:
            logger.error("Failed to initialize model from disk: %s", str(e))
            raise

    def recommend_for_user(self, user_id: int, num_recs: int = 10) -> Dict[str, Any]:
        """
        Generates top-N recommendations for a registered user 
        by retrieving pre-calculated latent factors.
        """
        if self.user_categories is None or user_id not in self.user_categories:
            return {"error": "User ID not found (Cold Start protection)."}
            
        try:
            # Translate external ID to internal matrix index
            u_idx = self.user_categories.get_loc(user_id)
            
            # Execute recommendation using internal ALS vectors
            ids, _ = self.model.recommend(
                userid=u_idx, 
                user_items=self.sparse_matrix[u_idx], 
                N=num_recs,
                filter_already_liked_items=True
            )
            
            # Map indices back to external product identifiers
            recommended_pids = [int(self.product_categories[p_idx]) for p_idx in ids]
            
            return {"user_id": int(user_id), "recommendations": recommended_pids}
        except Exception as e:
            logger.error("Inference error for user %s: %s", user_id, str(e))
            return {"error": "Internal recommendation engine failure."}

    def get_similar_items(self, item_id: int, num_recs: int = 10) -> Dict[str, Any]:
        """
        Identifies alternative products based on cosine similarity 
        within the latent item-feature space.
        """
        if self.product_categories is None or item_id not in self.product_categories:
            return {"error": "Target Product ID not found in system."}
            
        try:
            p_idx = self.product_categories.get_loc(item_id)
            
            # Retrieve N+1 items to allow for self-exclusion
            ids, _ = self.model.similar_items(itemid=p_idx, N=num_recs + 1)

            # Filter out the source item to prevent circular recommendations
            recommended_pids = [
                int(self.product_categories[idx]) for idx in ids if idx != p_idx
            ]

            return {
                "item_id": int(item_id), 
                "similar_items": recommended_pids[:num_recs]
            }
        except Exception as e:
            logger.error("Similarity calculation failed for product %s: %s", item_id, str(e))
            return {"error": "Internal similarity engine failure."}

    def recommend_dynamic(self, item_ids: List[int], num_recs: int = 10) -> Dict[str, Any]:
        """
        Real-time session-based recommendations. 
        Calculates a temporary user factor on-the-fly for new users (Cold Start).
        """
        if self.model is None or self.product_categories is None:
            return {"error": "Inference engine components not initialized."}
            
        try:
            # Filter and map provided IDs to internal indices
            valid_indices = [
                self.product_categories.get_loc(iid) 
                for iid in item_ids if iid in self.product_categories
            ]
            
            if not valid_indices:
                return {"error": "None of the provided product IDs exist in the current model."}
            
            # Construct a transient sparse user-item interaction vector
            num_items = len(self.product_categories)
            data = np.ones(len(valid_indices), dtype=np.float32)
            rows = np.zeros(len(valid_indices), dtype=np.int32)
            cols = np.array(valid_indices, dtype=np.int32)
            user_vector = csr_matrix((data, (rows, cols)), shape=(1, num_items))

            # Trigger real-time re-calculation of user factors (Analytical Cold Start)
            ids, _ = self.model.recommend(
                userid=0, 
                user_items=user_vector,
                N=num_recs,
                recalculate_user=True,
                filter_already_liked_items=True
            )
            
            recommended_items = [int(self.product_categories[idx]) for idx in ids]
            
            return {
                "input_item_ids": [int(i) for i in item_ids], 
                "recommendations": recommended_items
            }
            
        except Exception as e:
            logger.error("Dynamic session recommendation failed: %s", str(e))
            return {"error": "Real-time inference error occurred."}