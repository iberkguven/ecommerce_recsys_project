from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.recommender import ALSRecommender
import pickle
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Pydantic Schemas for Data Validation ---
class DynamicRecommendRequest(BaseModel):
    """Schema for cold-start / dynamic recommendation requests."""
    item_ids: List[int]
    num_recs: Optional[int] = 10

# --- Global State Management ---
recommender = ALSRecommender()
product_lookup = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager to handle startup and shutdown events.
    Loads the ML model and lookup table into RAM once.
    """
    global product_lookup
    try:
        # Load the serialized ALS model artifacts
        recommender.load_model("models/als_model_best.pkl")
        
        # Load product metadata for enrichment (brand, category, price)
        with open("models/product_lookup.pkl", "rb") as f:
            product_lookup = pickle.load(f)
            
        logger.info("Service Startup: Model and Lookup Table loaded successfully.")
    except Exception as e:
        logger.error(f"Service Startup Failure: {str(e)}")
        raise e
    yield
    logger.info("Service Shutdown: Cleaning up resources.")

# Initialize FastAPI App with Lifespan
app = FastAPI(
    title="E-Commerce Recommender API",
    description="High-performance ALS-based recommendation engine for e-commerce.",
    version="2.0.0",
    lifespan=lifespan
)

# --- Internal Helper Functions ---
def get_product_details(pid_list: List[int]) -> List[Dict[str, Any]]:
    """
    Enriches a list of Product IDs with metadata from the lookup table.
    Ensures consistent key naming (product_id) to prevent UI errors.
    """
    details = []
    if product_lookup is None:
        logger.warning("Product lookup table not initialized yet.")
        return details
        
    for pid in pid_list:
        if pid in product_lookup.index:
            item = product_lookup.loc[pid]
            details.append({
                "product_id": int(pid),
                "brand": str(item.get('brand', 'Unknown')),
                "category": str(item.get('category_code', 'N/A')),
                "price": float(item.get('price', 0.0))
            })
    return details

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    """Service health check endpoint."""
    return {"status": "healthy", "model_loaded": recommender.model is not None}

@app.get("/recommend/{user_id}")
async def get_user_recommendations(user_id: int, n: int = 10):
    """
    Standard Collaborative Filtering: 
    Predicts items for existing users based on historical interaction.
    """
    result = recommender.recommend_for_user(user_id, num_recs=n)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    enriched_items = get_product_details(result["recommendations"])
    
    return {
        "status": "success",
        "user_id": user_id,
        "results_count": len(enriched_items),
        "items": enriched_items
    }

@app.get("/similar/{product_id}")
async def get_similar_items(product_id: int, n: int = 10):
    """
    Item-to-Item Similarity:
    Finds alternative products using vector distance in the latent space.
    """
    result = recommender.get_similar_items(product_id, num_recs=n) 

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    enriched_items = get_product_details(result["similar_items"])

    return {
        "status": "success",
        "base_product_id": product_id,
        "results_count": len(enriched_items),
        "items": enriched_items
    }

@app.post("/recommend_dynamic")
async def get_dynamic_recommendations(request: DynamicRecommendRequest):
    result = recommender.recommend_dynamic(request.item_ids, num_recs=request.num_recs)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    enriched_items = get_product_details(result["recommendations"])
    
    input_categories = []
    for pid in request.item_ids:
        if pid in product_lookup.index:
            input_categories.append(str(product_lookup.loc[pid, 'category_code']).split('.')[0])

    boosted_items = sorted(
        enriched_items, 
        key=lambda x: any(cat in str(x['category']) for cat in input_categories), 
        reverse=True
    )

    return {
        "status": "success",
        "items": boosted_items
    }
@app.get("/top_products")
async def get_top_products(n: int = 50):
    """Returns diverse popular products for the UI selection menu."""
    if product_lookup is None:
        return {} 
    
    # Variety filtering logic
    top_items = product_lookup.head(n * 3).drop_duplicates(subset=['category_code', 'brand']).head(n)
    
    result = {}
    for pid, row in top_items.iterrows():
        cat = str(row['category_code']).split('.')[-1].capitalize()
        brand = str(row['brand']).upper()
        display_name = f"{cat} - {brand}"
        result[display_name] = int(pid)
    return result