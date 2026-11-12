# src/inspect_data.py
import pandas as pd
import os

def inspect():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'practice', 'questions.csv')
    df = pd.read_csv(csv_path, dtype=str)
    
    print("Columns:", df.columns.tolist())
    
    # Filter to rows that actually have gold evidence
    has_gold = df[df['gold_evidence'].notna() & (df['gold_evidence'] != '')]
    
    print("\nSample gold_evidence formats:")
    print(has_gold['gold_evidence'].head(5).tolist())
    
    unanswerable_count = len(df) - len(has_gold)
    print(f"\nTotal questions: {len(df)}")
    print(f"Unanswerable rows (empty gold_evidence): {unanswerable_count}")

if __name__ == "__main__":
    inspect()