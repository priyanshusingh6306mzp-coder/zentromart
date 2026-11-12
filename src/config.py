# src/config.py
from dotenv import load_dotenv
load_dotenv()

import os
import sys
from pathlib import Path

# ---------- dataset location (one place) ----------
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.environ.get("ZM_DATASET_DIR", ROOT))   # folder that contains kb/, data/, tools/
KB_DIR = DATASET_DIR / "kb"
DATA_DIR = DATASET_DIR / "data"
PRACTICE_DIR = DATASET_DIR / "practice"
INDEX_DIR = Path(os.environ.get("ZM_INDEX_DIR", ROOT / ".index"))   # generated, git-ignored

if str(DATASET_DIR) not in sys.path:
    sys.path.append(str(DATASET_DIR))


def check_dataset():
    if not KB_DIR.is_dir() or not DATA_DIR.is_dir() or not (DATASET_DIR / "tools" / "tools.py").is_file():
        raise SystemExit(f"Dataset not found at {DATASET_DIR}. "
                         "Set ZM_DATASET_DIR to the folder containing kb/, data/ and tools/.")


# ---------- models ----------
ROUTER_MODEL = "qwen/qwen3.8-27b"
ANSWER_MODEL = "qwen/qwen3.8-27b"
LLM_EXTRA = {}   # optional extra Groq parameters; leave empty unless verified for your model

# The Capstone requires simulating this exact date
TODAY = "2026-10-20"

# ---------- Step 1 baseline prompt ----------
SYSTEM_PROMPT = """You are a customer support AI assistant for ZentroMart, an online retailer for groceries and electronics in India.

CRITICAL RULES:
1. Introduction: In your very first reply of a conversation, you must state that you are an AI assistant.
2. Tone & Length: Keep replies polite, highly concise, and conversational (max 2-3 short sentences).
3. No Hallucinations: Never invent or assume store policies, delivery dates, refund amounts, or order details.
4. Unknowns: If you do not know the answer based on provided context, explicitly state "I don't know" and offer to connect them to a human agent.
"""

# ---------- Step 2 RAG prompt ----------
RAG_SYSTEM_PROMPT = """You are a customer support AI assistant for ZentroMart, an online retailer for groceries and electronics in India.

CRITICAL RULES:
1. Introduction: In your very first reply of a conversation, you state you are an AI.
2. Tone & Length: Polite, highly concise, max 2-3 short spoken sentences.
3. Grounding: Answer ONLY using the facts in the POLICY CONTEXT block below.
4. Unknowns: If the context does not contain the answer, explicitly state "I don't know" and offer a human handover. NEVER invent policies, dates, or amounts.

POLICY CONTEXT:
{context}
"""