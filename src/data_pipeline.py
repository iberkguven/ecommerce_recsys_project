import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from typing import Optional

class DataPipeline:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.user_categories = None
        self.product_categories = None
        self.optimized_dtypes = {
            'event_type': 'category',
            'product_id': 'int32',
            'category_id': 'int64',
            'category_code': 'category',
            'brand': 'category',
            'price': 'float32',
            'user_id': 'int32'
        }

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path, dtype=self.optimized_dtypes)
        return df.dropna(subset=['user_id', 'product_id'])

    def create_product_lookup(self, df: pd.DataFrame) -> pd.DataFrame:
        lookup = df[['product_id', 'category_code', 'brand', 'price']].drop_duplicates('product_id')
        return lookup.set_index('product_id')

    def process_and_build_matrix(self, df: pd.DataFrame) -> csr_matrix:
        df['user_id'] = df['user_id'].astype('category')
        df['product_id'] = df['product_id'].astype('category')

        self.user_categories = df['user_id'].cat.categories
        self.product_categories = df['product_id'].cat.categories

        matrix = csr_matrix((
            np.ones(len(df), dtype=np.int8),
            (df['user_id'].cat.codes, df['product_id'].cat.codes)
        ))
        return matrix