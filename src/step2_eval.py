# src/step2_eval.py
import os
import pandas as pd
from agent import ask_policy
from llm import chat
from config import SYSTEM_PROMPT # We reuse the same judge prompt!

JUDGE_PROMPT = """You are an impartial evaluator grading a customer service AI.
Compare the 'AI Answer' against the 'Expected Decision'.
Did the AI reach the correct expected decision? 
Reply ONLY with "PASS" or "FAIL". Do not provide any other text."""

def calculate_metrics(retrieved_ids, gold_list, k_values=[3, 5, 8]):
    """Calculates hit@k and Reciprocal Rank."""
    metrics = {}
    
    # Clean gold list to just KB section IDs (ignore CSV references like products.csv:ELE-001)
    gold_kb_ids = [g.strip() for g in gold_list if '.csv' not in g]
    if not gold_kb_ids:
        return {f"hit@{k}": 0 for k in k_values}, 0.0 # No KB evidence required
        
    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(retrieved_ids, 1):
        # Check if the chunk ID ends with any of our gold section IDs (e.g., WAR-2)
        if any(gold in chunk_id for gold in gold_kb_ids):
            reciprocal_rank = 1.0 / rank
            break # Found the first relevant chunk
            
    for k in k_values:
        top_k_ids = retrieved_ids[:k]
        hit = 1 if any(any(gold in cid for gold in gold_kb_ids) for cid in top_k_ids) else 0
        metrics[f"hit@{k}"] = hit
        
    return metrics, reciprocal_rank

def evaluate_step2():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'practice', 'questions.csv')
    df = pd.read_csv(csv_path, dtype=str)
    
    results = []
    passed = 0
    total_mrr = 0.0
    total_hit5 = 0
    answerable_count = 0

    print("Running Step 2 (RAG) Evaluation...")
    for index, row in df.iterrows():
        question = row['question'] # We'll use clean question first. Try 'asr_text' later.
        expected = row.get('expected_decision', 'Unknown')
        gold_raw = str(row.get('gold_evidence', ''))
        
        # Run RAG Agent
        ai_answer, chunk_ids = ask_policy(question)
        
        # LLM Judge
        judge_messages = [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"Expected Decision: {expected}\n\nAI Answer: {ai_answer}"}
        ]
        verdict = chat(judge_messages).strip().upper()
        if "PASS" in verdict:
            passed += 1
            
        # Retrieval Metrics
        if gold_raw.strip() and gold_raw.lower() != 'nan':
            answerable_count += 1
            gold_list = gold_raw.split(';')
            metrics, mrr = calculate_metrics(chunk_ids, gold_list)
            total_mrr += mrr
            total_hit5 += metrics['hit@5']
        else:
            metrics, mrr = {}, 0.0
            
        results.append({
            "question": question,
            "expected_decision": expected,
            "step2_answer": ai_answer,
            "verdict": verdict,
            "retrieved_chunks": chunk_ids,
            "mrr": mrr
        })
        print(f"Q{index+1}: {verdict} | MRR: {mrr:.2f}")

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv('../eval_results/step2_answers.csv', index=False)
    
    # Final Metrics
    avg_mrr = total_mrr / answerable_count if answerable_count else 0
    hit_rate = total_hit5 / answerable_count if answerable_count else 0
    
    print(f"\n=== STEP 2 RESULTS ===")
    print(f"Decision Score: {passed} / {len(df)}")
    print(f"Retrieval Hit@5: {hit_rate*100:.1f}%")
    print(f"Retrieval MRR: {avg_mrr:.3f}")

if __name__ == "__main__":
    evaluate_step2()