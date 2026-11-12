# src/eval.py
import pandas as pd
from llm import chat
from config import SYSTEM_PROMPT

JUDGE_PROMPT = """You are an impartial evaluator grading a customer service AI.
Compare the 'AI Answer' against the 'Expected Decision'.
Did the AI reach the correct expected decision? 
Reply ONLY with "PASS" or "FAIL". Do not provide any other text."""

def evaluate_baseline():
    # dtype=str is required so Pandas doesn't drop leading zeros in order IDs!
    df = pd.read_csv('../practice/questions.csv', dtype=str)
    
    results = []
    passed = 0

    print("Running baseline evaluation on Groq...")
    for index, row in df.iterrows():
        question = row['question']
        expected = row.get('expected_decision', 'Unknown')
        
        # 1. Generate the baseline answer
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        ai_answer = chat(messages)
        
        # 2. Score the answer using Groq as a judge
        judge_messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Expected Decision: {expected}\n\nAI Answer: {ai_answer}"}
        ]
        verdict = chat(judge_messages).strip().upper()
        
        if "PASS" in verdict:
            passed += 1
            
        results.append({
            "question": question,
            "expected_decision": expected,
            "step1_answer": ai_answer,
            "verdict": verdict
        })
        print(f"Q{index+1}: {verdict}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('../eval_results/step1_answers.csv', index=False)
    
    print(f"\nStep 1 Baseline Score: {passed} / {len(df)}")
    print("Results saved to eval_results/step1_answers.csv")

if __name__ == "__main__":
    evaluate_baseline()