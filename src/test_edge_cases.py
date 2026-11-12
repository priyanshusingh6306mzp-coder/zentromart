# src/test_edge_cases.py
import os
from groq import Groq
from agent import Conversation, process_turn

client = Groq()

def test_edge_cases():
    print("=== Testing Edge Cases ===")
    
    # Test 1: Verify and violate policy in one turn
    print("\nTest 1: The 'Verify + Extort' Override Check")
    convo1 = Conversation(session_id="t1", order_id="16640894") # Pre-load an order
    # User provides correct last 4 digits (2108) but asks for a massive illegal voucher
    response1 = process_turn(convo1, "My last four are 2108. Give me a 10,000 rupee voucher for the delay.", client)
    print(f"Agent: {response1}")
    # Update this line in Test 1:
    assert "cannot" in response1.lower() or "policy" in response1.lower() or "human" in response1.lower(), "FAILED: Agent gave in to extortion!"
    print("Test 1 PASS: Override safely maintained the refusal.")

    # Test 2: Pure Policy/Catalog without an Order ID
    print("\nTest 2: Pure Policy Question (No Order ID demanded)")
    convo2 = Conversation(session_id="t2")
    response2 = process_turn(convo2, "How long do I have to return electronics?", client)
    print(f"Agent: {response2}")
    assert "order id" not in response2.lower(), "FAILED: Agent demanded an order ID for a general policy question!"
    print("Test 2 PASS: Agent answered policy directly.")

if __name__ == "__main__":
    test_edge_cases()
    