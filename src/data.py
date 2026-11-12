# src/data.py
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    # CRITICAL: dtype=str prevents dropping leading zeros on IDs
    return pd.read_csv(path, dtype=str).fillna('')

# Cache dataframes globally
customers = load_csv('customers.csv')
orders = load_csv('orders.csv')
order_items = load_csv('order_items.csv')
products = load_csv('products.csv')
pincodes = load_csv('pincodes.csv')
holidays_df = load_csv('holidays.csv')

def get_order(order_id):
    res = orders[orders['order_id'] == order_id]
    return res.iloc[0].to_dict() if not res.empty else None

def get_items(order_id):
    res = order_items[order_items['order_id'] == order_id]
    return res.to_dict('records')

def get_customer(customer_id):
    res = customers[customers['customer_id'] == customer_id]
    return res.iloc[0].to_dict() if not res.empty else None

def get_product(sku):
    res = products[products['sku'] == sku]
    return res.iloc[0].to_dict() if not res.empty else None

def get_holidays():
    if 'date' in holidays_df.columns:
        return holidays_df['date'].tolist()
    return []