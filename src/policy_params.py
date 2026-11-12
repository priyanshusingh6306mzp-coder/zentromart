# src/policy_params.py

# Update just this dictionary in src/policy_params.py
POLICY_PARAMS = {
    "return_windows_calendar_days": {
        "electronics": 7,     
        "grocery": 2,       
        "books_media": 10,
        "fashion": 15,
        "default": 0 # Non-returnable fallback if category is unknown
    },
    # ... keep the rest the same

    # Business days per [REF-1], [REF-2] in pol_refunds.md
    "refund_sla_business_days": {
        "wallet": 0, # Instant / 2 hours [REF-2]
        "upi": 3,    # 1-3 business days [REF-2]
        "net_banking": 5, # 3-5 business days [REF-2]
        "credit_card": 7  # 5-7 business days [REF-2]
    },
    "delivery_sla_days": 3         # [DEL-1]
}