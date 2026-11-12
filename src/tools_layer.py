# src/tools_layer.py
import data
import dates
# In your actual environment, you will import the provided tools:
# from tools.tools import lookup_order, create_ticket

def verify_caller(order_id, phone_last4=None, pincode=None):
    """Verifies user identity based on Capstone rules."""
    order = data.get_order(order_id)
    if not order:
        return False, "not_found"
    
    customer = data.get_customer(order['customer_id'])
    
    if phone_last4 and customer and customer.get('phone', '').endswith(phone_last4):
        return True, ""
    if pincode and order.get('delivery_pincode') == pincode:
        return True, ""
        
    return False, "verification_failed"

def safe_lookup_order(order_id, phone_last4=None, pincode=None):
    if not phone_last4 and not pincode:
        return {"ok": False, "reason": "needs_verification", "order": None}
    
    is_verified, reason = verify_caller(order_id, phone_last4, pincode)
    if not is_verified:
        return {"ok": False, "reason": reason, "order": None}
        
    return {"ok": True, "reason": "", "order": data.get_order(order_id)}

def order_facts(order_id, phone_last4=None, pincode=None):
    """
    Computes facts purely in code. 
    Pass this dictionary into your LLM prompt so it doesn't do date math!
    """
    lookup = safe_lookup_order(order_id, phone_last4, pincode)
    if not lookup['ok']:
        return {"error": lookup['reason']}
        
    order = lookup['order']
    items = data.get_items(order_id)
    
    # Treat returned statuses as having been delivered/shipped at some point
    is_delivered = order.get("status") in ["delivered", "return_picked_up", "return_refund_initiated", "returned_refunded"]
    is_shipped = is_delivered or order.get("status") == "shipped"
    
    facts = {
        "order_id": order_id,
        "status": order.get("status"),
        "shipped": is_shipped,
        "delivered": is_delivered,
        "days_late": dates.days_late(order.get("promised_date")),
        "items": []
    }
    
    for item in items:
        # CRITICAL FIX: Pull return_category directly from order_items
        raw_cat = item.get("return_category", "").strip().lower()
        cat = raw_cat if raw_cat else "default"
        
        facts["items"].append({
            "sku": item.get("sku"),
            "category": cat,
            "within_return_window": dates.is_within_return_window(order.get("delivery_date"), cat) if is_delivered else False
        })
        
    return facts

def product_info(sku):
    """Catalog lookup: No verification required."""
    prod = data.get_product(sku)
    if not prod:
        return {"error": "product_not_found"}
    return prod

def pincode_info(pincode):
    """Delivery zone lookup: No verification required."""
    # Assuming data.pincodes is a DataFrame loaded in data.py
    if data.pincodes.empty:
        return {"error": "pincode_data_missing"}
    res = data.pincodes[data.pincodes['pincode'] == pincode]
    if res.empty:
        return {"error": "pincode_not_found"}
    return res.iloc[0].to_dict()

def safe_create_ticket(summary):
    """
    Wrapper for create_ticket. 
    Rule: Never report a handover as done unless created=True.
    """
    # result = create_ticket(summary) # Call the real tool here
    
    # Mocking real tool response for now
    result = {"created": True, "ticket_id": "TKT-999", "error": ""}
    return result