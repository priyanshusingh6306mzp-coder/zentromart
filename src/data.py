# src/data.py
import pandas as pd

from config import DATA_DIR


def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        print(f"WARNING: {path} not found")
        return pd.DataFrame()
    # dtype=str keeps leading zeros on IDs
    return pd.read_csv(path, dtype=str).fillna("")


customers = load_csv("customers.csv")
orders = load_csv("orders.csv")
order_items = load_csv("order_items.csv")
products = load_csv("products.csv")
pincodes = load_csv("pincodes.csv")
holidays_df = load_csv("holidays.csv")


def _one(df, col, value):
    if df.empty or col not in df.columns:
        return None
    res = df[df[col] == value]
    return res.iloc[0].to_dict() if not res.empty else None


def get_order(order_id):
    return _one(orders, "order_id", order_id)


def get_items(order_id):
    if order_items.empty or "order_id" not in order_items.columns:
        return []
    return order_items[order_items["order_id"] == order_id].to_dict("records")


def get_customer(customer_id):
    return _one(customers, "customer_id", customer_id)


def get_product(sku):
    return _one(products, "sku", sku)


def get_holidays():
    if "date" in holidays_df.columns:
        return holidays_df["date"].tolist()
    return []