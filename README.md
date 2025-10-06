# LLM Benchmarking Project

This repository contains my project for benchmarking Large Language Model (LLM) agents across research-oriented tasks. The focus is on evaluating how different agents—ranging from single-tool to multi-tool configurations—perform in a structured, reproducible, and safety-conscious environment.

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

✔️ **Milestone 2: Data Preparation & Model Choice** – Inprogress.

🔜 **Milestone 3: Implementation & Experiments** - (To be continued...)

🔜 **Milestone 4: Final Report & Presentation** - (To be continued...)

---

## Repository Structure

```
/data         -> Benchmark tasks (JSON format)  
/agents       -> Single-tool and multi-tool agent definitions  
/evaluation   -> Metric calculation scripts  
/safety       -> Adversarial test cases and sandbox configs  
README.md     -> Project overview and milestones  
```

## Goals

This project aims to provide a reproducible benchmark that not only measures performance but also stress-tests safety and robustness, making it valuable for both research and practical deployment scenarios.


## How to run
(To be continued...)
