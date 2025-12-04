<p align="center">
  <img src="openai.svg" alt="OpenAI" width="30" height="30" style="vertical-align:middle; margin-right:8px;">
  <strong>VS</strong>
  <img src="Google_Gemini_icon_2025.svg.png" alt="Google Gemini" width="30" height="30" style="vertical-align:middle;">
</p>

# Auto Bench: A Comprehensive Benchmark Suite for Evaluating Multi-Tool Agentic LLMs 

This repository contains my project for benchmarking Large Language Model (LLM) agents across research-oriented tasks. The focus is on evaluating how different agents—ranging from single-tool to multi-tool configurations—perform in a structured, reproducible, and safety-conscious environment.

![Benchmark Evaluation Pipeline](benchmark_evaluation_pipeline.svg)
## Project Overview

* **Domain Focus:** Public Health & Pandemics (chosen for its rich, data-driven context).
* **LLM Backends:** GPT-4o mini (OpenAI) and Gemini flash (Google).  


* **Agents:**

  * *Single-Tool Agent* → Web searcher.
  * *Multi-Tool Agent* → Content extractor + quiz generator.
* **Benchmark Structure:** 40 JSON-based tasks with `gold_end_state` and `oracle_plan` definitions.
* **Evaluation Metrics:** Task Success Rate, Tool-Call Precision/Recall, Plan Efficiency, and Cost per Successful Task.
* **Safety & Robustness Plan:** Includes sandboxed execution, Git-based tracking, and adversarial tests (prompt injection, tool misuse, data exfiltration).

## Current Status

✔️ **Milestone 1: Proposal & Design** – Completed.  

✔️ **Milestone 2: Data Preparation & Model Choice** – Completed.

✔️ **Milestone 3: Implementation & Experiments** - Completed.

✔️ **Milestone 4: Final Report & Presentation** - Completed.

---

## Repository Structure

```
│
├── agents.py                  # No-Tool, Single-Tool, and Multi-Tool Agent implementations
├── run_benchmark.py           # Main execution engine for running agents on benchmark scenarios
├── evaluation.py              # Computes performance metrics from agent logs
├── test_functions.py          # Pytest suite with mocked LLM calls (zero API cost)
│
├── benchmark_tasks/           # 40 curated JSON benchmark tasks
├── knowledge_base/            # Static documents used for internal RAG search
│
├── requirements.txt           # Required Python dependencies
└── README.md                  # Project documentation
```

## Goals

This project aims to provide a reproducible benchmark that not only measures performance but also stress-tests safety and robustness, making it valuable for both research and practical deployment scenarios.


## How to run
## **1. Environment Setup**

Open a terminal inside the project folder and run:

```bash
pip install -r requirements.txt
```

This installs all required libraries, including:

* `openai`
* `google-generativeai`
* `pytest`
* `python-dotenv`

---

## **2. Configure API Keys (Required for Full Benchmark)**

The benchmark uses both **OpenAI GPT-4.1 Mini** and **Gemini Flash**.

Create a `.env` file in the root directory:

```txt
OPENAI_API_KEY=your-openai-key-here
GOOGLE_API_KEY=your-google-gemini-key-here
```

Without keys:

* ✔ Tests will still run (mocked)
* ✘ Full benchmark will fail

---

## **3. Run Logic Tests (No API Cost)**

This validates your agent pipeline, tool routing, and RAG system **without spending tokens**.

```bash
pytest test_functions.py
```

What the tests verify:

| Component         | Validation                                    |
| ----------------- | --------------------------------------------- |
| No-Tool Agent     | Basic LLM call + logging structure            |
| Single-Tool Agent | ReAct parsing + web search + final answer     |
| Multi-Tool Agent  | Correct routing to quiz generator & extractor |
| Web Search        | Expected matching documents returned          |

---

## **4. Run the Full Benchmark**

(Requires API keys)

```bash
python run_benchmark.py
```

Runtime: **10–15 minutes**

Output:

```
final_benchmark_results.json
```

This file contains:

* Agent decisions & logs
* Token usage
* Tool calls
* Raw LLM responses

---

## **5. Generate Metrics & Final Report**

After running the benchmark:

```bash
python evaluation.py
```

Produces:

```
milestone_4_final_report.csv
```

