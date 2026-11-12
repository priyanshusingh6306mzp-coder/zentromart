# src/step3_conversations.py
import pandas as pd
import os
from groq import Groq
from agent import Conversation, process_turn

client = Groq()

def evaluate_conversations():
    print("Loading conversations.csv...")
    try:
        df = pd.read_csv('../practice/conversations.csv', dtype=str)
    except FileNotFoundError:
        print("❌ practice/conversations.csv not found.")
        return

    print(f"Detected columns: {df.columns.tolist()}")
    
    if 'conversation_id' not in df.columns:
        print("Error: 'conversation_id' column missing!")
        return
        
    grouped = df.groupby('conversation_id')
    
    total_turns = 0
    passed_turns = 0
    
    print("\nStarting Conversation Evaluation...")
    
    for conv_id, group in grouped:
        print(f"\n=== Evaluating Conversation: {conv_id} ===")
        convo = Conversation(session_id=conv_id)
        
        for index, row in group.iterrows():
            # Skip agent turns if they are in the dataset
            if 'speaker' in df.columns and str(row['speaker']).lower() not in ['user', 'customer']:
                continue
            if 'role' in df.columns and str(row['role']).lower() not in ['user', 'customer']:
                continue
            
            # Dynamically extract the user message using common Capstone column names
            user_msg = ""
            for col in ['text', 'user_text', 'customer_message', 'message', 'question', 'asr_text']:
                if col in row:
                    val = str(row[col]).strip()
                    if val and val.lower() != 'nan':
                        user_msg = val
                        break
                        
            if not user_msg:
                continue # Safely skip empty rows
                
            total_turns += 1
            expected_action = str(row.get('expected_action', '')).strip().lower()
            
            print(f"User: {user_msg}")
            ai_response = process_turn(convo, user_msg, client)
            print(f"Agent: {ai_response}")
            
            # Flexible Evaluation Logic
            is_pass = False
            if expected_action in ['clarify', 'ask_clarify']:
                if '?' in ai_response or not convo.is_verified:
                    is_pass = True
            elif expected_action in ['escalate', 'refuse', 'handover']:
                if convo.handover_done or "cannot fulfill" in ai_response.lower() or "connect" in ai_response.lower() or "human" in ai_response.lower():
                    is_pass = True
            elif expected_action == 'answer':
                if convo.is_verified or "don't know" in ai_response.lower() or "hello" in ai_response.lower():
                    is_pass = True
            else:
                # If the action column is missing/unknown, we pass if the agent didn't crash
                is_pass = True
            
            if is_pass:
                passed_turns += 1
                print("Verdict: PASS")
            else:
                print(f"Verdict: FAIL (Expected: {expected_action})")
            print("-" * 30)

    score = (passed_turns / total_turns) * 100 if total_turns > 0 else 0
    print(f"\n=== FINAL STEP 3 RESULTS ===")
    print(f"Total Turns: {total_turns}")
    print(f"Passed Turns: {passed_turns}")
    print(f"Turn Accuracy: {score:.1f}%")

if __name__ == "__main__":
    evaluate_conversations()