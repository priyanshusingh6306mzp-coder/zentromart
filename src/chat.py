# src/chat.py
import os
from groq import Groq
from agent import Conversation, process_turn

# Initialize your Groq client
client = Groq()

def start_chat():
    print("=== ZentroMart VoiceOps Interactive Tester ===")
    print("Type 'quit' to exit.\n")
    
    # Initialize a fresh state for this session
    convo = Conversation(session_id="session_001")
    
    while True:
        user_input = input("Customer: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        # Pass the input through our state machine
        ai_response = process_turn(convo, user_input, client)
        
        print(f"\nAgent: {ai_response}\n")
        
        # Print the internal state so you can watch the agent "think"
        print(f"[Debug State] Order: {convo.order_id} | Pin: {convo.pincode} | Verified: {convo.is_verified} | Handover: {convo.handover_done}")
        print("-" * 50)

if __name__ == "__main__":
    start_chat()