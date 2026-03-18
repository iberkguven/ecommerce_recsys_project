import streamlit as st
import requests
import pandas as pd
import os

# System Configuration
# Uses service name 'api' as defined in docker-compose
API_BASE_URL = os.getenv("API_URL", "http://api:8000")
st.set_page_config(page_title="ShopSmart AI | Recommender", layout="wide")

# --- DATA LOADING FUNCTIONS ---

@st.cache_data(ttl=3600) # Cache catalog for 1 hour
def load_dynamic_catalog():
    """Fetches popular products from API to populate the selection menu."""
    try:
        # We call the new endpoint we planned
        response = requests.get(f"{API_BASE_URL}/top_products?n=50", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.sidebar.error(f"Catalog Load Error: {e}")
    
    # Static Fallback if API is not ready or fails
    return {
        "Smartphone - Apple iPhone 13": 1002522,
        "Smartphone - Xiaomi Redmi": 1004186,
        "TV - Samsung Crystal UHD": 1004187,
        "Vacuum - Dyson V11": 3601438,
        "Shoes - Nike Air Max": 28719069
    }

# --- UI HELPER FUNCTIONS ---

def render_product_cards(items: list):
    """Renders product metadata in a clean, grid-based layout."""
    if not items:
        st.warning("No recommendations found.")
        return

    # Use 5 columns for better space management with N=50
    cols_per_row = 5
    for i in range(0, len(items), cols_per_row):
        row_items = items[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for idx, item in enumerate(row_items):
            with cols[idx]:
                with st.container(border=True):
                    # Clean metadata display
                    st.markdown(f"**{item.get('brand', 'N/A').upper()}**")
                    st.caption(f"{item.get('category', 'N/A').split('.')[-1]}")
                    st.markdown(f"**${item.get('price', 0.0):.2f}**")
                    st.caption(f"ID: {item.get('product_id', 'N/A')}")

# --- MAIN APPLICATION ---

def main():
    st.title("🚀 ShopSmart AI | Enterprise Recommendation Engine")
    st.markdown("---")

    tab_user, tab_similar, tab_dynamic = st.tabs([
        "👤 Registered User", 
        "🔗 Content Similarity", 
        "🛒 Real-time Session (Cold Start)"
    ])

    # --- TAB 1: REGISTERED USER ---
    with tab_user:
        st.subheader("Personalized for You")
        u_id = st.number_input("Enter User ID", value=557746614, key="user_in")
        n_user = st.slider("Recommendations count", 10, 50, 20, key="n_user")
        
        if st.button("Generate Profile", type="primary", key="btn_user"):
            with st.spinner("Fetching user history..."):
                res = requests.get(f"{API_BASE_URL}/recommend/{u_id}?n={n_user}")
                if res.status_code == 200:
                    render_product_cards(res.json()['items'])
                else:
                    st.error("User profile analysis failed.")

    # --- TAB 2: SIMILAR PRODUCTS ---
    with tab_similar:
        st.subheader("Discovery Mode")
        p_id = st.number_input("Enter Product ID", value=1004186, key="prod_in")
        n_sim = st.slider("Discovery depth", 10, 50, 20, key="n_sim")
        
        if st.button("Find Similar Items", type="primary", key="btn_sim"):
            with st.spinner("Analyzing product DNA..."):
                res = requests.get(f"{API_BASE_URL}/similar/{p_id}?n={n_sim}")
                if res.status_code == 200:
                    render_product_cards(res.json()['items'])
                else:
                    st.error("Product similarity engine error.")

    # --- TAB 3: DYNAMIC COLD START ---
    with tab_dynamic:
        st.subheader("Instant Cart Personalization")
        st.write("Simulate a shopping session to get real-time suggestions.")
        
        # Loading catalog from API or Fallback
        catalog = load_dynamic_catalog()
        
        selected_names = st.multiselect(
            "Add products to your cart:", 
            options=list(catalog.keys()),
            help="Select products from various categories for better cross-selling results."
        )
        
        n_dyn = st.slider("Inference Density", 10, 50, 20, key="n_dyn")

        if st.button("Recalculate Style Profile", type="primary", key="btn_dyn"):
            if not selected_names:
                st.warning("Your cart is empty. Please select products.")
            else:
                with st.spinner("Performing real-time matrix recalculation..."):
                    selected_ids = [int(catalog[name]) for name in selected_names]
                    payload = {"item_ids": selected_ids, "num_recs": n_dyn}
                    
                    res = requests.post(f"{API_BASE_URL}/recommend_dynamic", json=payload)
                    if res.status_code == 200:
                        render_product_cards(res.json()['items'])
                    else:
                        st.error("Dynamic session inference failed.")

if __name__ == "__main__":
    main()