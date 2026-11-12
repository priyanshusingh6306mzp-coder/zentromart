# src/verify_order_facts.py
import pandas as pd
import data
import tools_layer

def verify_practice_orders():
    print("Loading conversations.csv to extract order IDs...")
    try:
        convos = pd.read_csv('../practice/conversations.csv', dtype=str)
    except FileNotFoundError:
        print("❌ conversations.csv not found in practice folder.")
        return

    # Grab all valid order IDs from the database
    all_db_orders = set(data.orders['order_id'].dropna().tolist())
    
    # Robust extraction: dump the whole dataframe to a string to avoid KeyErrors
    text_corpus = convos.to_string()
    
    # Find any database order ID that appears anywhere in the conversations
    found_orders = [o for o in all_db_orders if o in text_corpus]
    
    print(f"Found {len(found_orders)} distinct orders mentioned in practice conversations.\n")
    
    for order_id in found_orders:
        order_record = data.get_order(order_id)
        if not order_record:
            continue
            
        # Bypass security legitimately by using the real pincode
        real_pincode = order_record.get('delivery_pincode')
        facts = tools_layer.order_facts(order_id, pincode=real_pincode)
        
        print(f"--- Order: {order_id} ---")
        print(f"Status: {facts.get('status')} | Late: {facts.get('days_late')} days")
        for item in facts.get('items', []):
            print(f"  SKU: {item['sku']} | Cat: {item['category']} | Returnable: {item['within_return_window']}")
        print()

if __name__ == "__main__":
    verify_practice_orders()