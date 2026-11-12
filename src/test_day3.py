# src/test_day3.py
import data
import dates
import tools_layer

def run_tests():
    print("=== TESTING DATE MATH ===")
    print(f"System TODAY is set to: {dates.TODAY.date()}")
    
    # 20 Oct 2026 is a Tuesday. 5 business days should skip Sunday (Oct 25) 
    # and land on Monday, Oct 26 (assuming no holidays in that window).
    test_date = dates.add_business_days("2026-10-20", 5)
    print(f"5 Business days from today: {test_date}")
    
    print("\n=== TESTING DATA LOADERS ===")
    if not data.orders.empty:
        # Grab the very first order dynamically so it doesn't crash
        sample_order = data.orders.iloc[0]['order_id']
        sample_pincode = data.orders.iloc[0]['delivery_pincode']
        print(f"Successfully loaded {len(data.orders)} orders.")
        print(f"Sample Order: {sample_order} | Pincode: {sample_pincode}")
        
        print("\n=== TESTING TOOLS LAYER ===")
        # Test 1: Wrong Pincode (Should fail verification)
        failed = tools_layer.order_facts(sample_order, pincode="000000")
        print(f"Expected Failure (Wrong Pincode): {failed}")
        
        # Test 2: Correct Pincode (Should return order facts)
        success = tools_layer.order_facts(sample_order, pincode=sample_pincode)
        print("\nExpected Success (Order Facts for LLM):")
        for key, value in success.items():
            print(f"  {key}: {value}")
            
    else:
        print("❌ FAILED: CSV data not found. Check your data/ folder.")

if __name__ == "__main__":
    run_tests()