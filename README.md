# ZentroMart VoiceOps AI Agent

## Overview
This project is a production-grade voice pipeline and AI assistant for ZentroMart. It integrates Speech-to-Text (STT) and Text-to-Speech (TTS) via the Gnani.ai API, featuring a secure state machine for strict policy adherence, edge-case handling, and order verification.


## Setup Instructions
1. Obtain your Gnani API Key from [app.gnani.ai/voice](https://app.gnani.ai/voice).
2. Copy `.env.example` to `.env` and insert your credentials:
   ```bash
   cp .env.example .env

   pip install -r requirements.txt

   *   **Independent Queries Mode (Default):** Processes each file as a new customer.
    ```bash
    python run.py --input_dir input --output_dir output
    ```
*   **Multi-Turn Conversation Mode:** Add `--conversation` and name your files `conv1_turn1.wav`, `conv1_turn2.wav`, etc., to share memory state across consecutive audio clips.
    ```bash
    python run.py --input_dir input --output_dir output --conversation
    ```