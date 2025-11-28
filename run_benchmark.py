import os
import json
from agents import NoToolAgent, SingleToolAgent, MultiToolAgent
import time

# ANALYSIS HELPER FUNCTIONS

def calculate_success(agent_log: dict, ground_truth: dict) -> bool:
    """Compares the agent's final answer to the gold end state."""
    agent_answer = agent_log.get("final_answer", "").lower()
    gold_answer = ground_truth["gold_end_state"].get("answer", 
                    ground_truth["gold_end_state"].get("summary", "")).lower()
    
    if not gold_answer:
        return False 
        
    keywords = gold_answer.split()[:3] 
    return all(kw in agent_answer for kw in keywords)

def calculate_plan_efficiency(agent_log: dict, ground_truth: dict) -> float:
    """Calculates the plan efficiency."""
    agent_steps = len(agent_log.get("steps", []))
    oracle_steps = len(ground_truth.get("oracle_plan", []))
    
    agent_tool_calls = len([s for s in agent_log["steps"] if s["type"] == "tool_call"])
    oracle_tool_calls = len(ground_truth["oracle_plan"])
    
    if oracle_tool_calls == 0 and agent_tool_calls == 0:
        return 1.0 
    if agent_tool_calls == 0 or oracle_tool_calls == 0:
        return 0.0 
        
    return min(1.0, oracle_tool_calls / agent_tool_calls)

# MAIN BENCHMARK RUNNER

def run_full_benchmark():
    print("Starting benchmark run...")
    
    # 1. Define the Models to Compare
    models_to_test = ["gemini-2.5-flash-preview-09-2025", "gpt-4o-mini"]
    
    task_directory = "benchmark_tasks"
    all_results = []

    try:
        task_files = [f for f in os.listdir(task_directory) if f.endswith('.json')]
        task_files.sort()
        if not task_files:
            print(f"Error: No task files found in '{task_directory}'.")
            return
    except FileNotFoundError:
        print(f"Error: Directory not found: '{task_directory}'")
        return

    # 2. Loop through Models -> Tasks -> Agents
    for model_name in models_to_test:
        print(f"\n" + "="*40)
        print(f"  TESTING MODEL: {model_name}")
        print("="*40)

        # Initialize agents for THIS model
        agents = {
            "No-Tool": NoToolAgent(model_name=model_name),
            "Single-Tool": SingleToolAgent(model_name=model_name),
            "Multi-Tool": MultiToolAgent(model_name=model_name)
        }

        for task_file in task_files:
            task_path = os.path.join(task_directory, task_file)
            
            with open(task_path, 'r') as f:
                task_data = json.load(f)
            
            # Tries 'prompt', if missing tries 'task_text'
            task_prompt = task_data.get("prompt", task_data.get("task_text"))
            
            # Tries 'task_id', if missing tries 'id'
            task_id = task_data.get("task_id", task_data.get("id"))

            if not task_prompt:
                print(f"Skipping {task_file}: Could not find 'prompt' or 'task_text' key.")
                continue

            # Default to Multi-Tool if not specified, or map specifically
            task_agent_type = task_data.get("agent_classification", "Multi-Tool")
            
            # Handles "No-Tool Agent" OR just "No-Tool"
            if "No-Tool" in task_agent_type: 
                agent_key = "No-Tool"
            elif "Single-Tool" in task_agent_type: 
                agent_key = "Single-Tool"
            else: 
                agent_key = "Multi-Tool"
            
            agent = agents[agent_key]
            
            print(f"\n--- Task: {task_id} | Model: {model_name} | Agent: {agent_key} ---")

            # Run the agent
            try:
                run_log = agent.run(task_prompt)
                
                # Calculate immediate metrics
                is_success = calculate_success(run_log, task_data)
                plan_efficiency = calculate_plan_efficiency(run_log, task_data)
                
                print(f"  -> Success: {is_success}, Efficiency: {plan_efficiency:.2f}")
                
                # Save Result
                result = {
                    "task_id": task_id,
                    "model": model_name,
                    "agent_type": agent_key,
                    "success": is_success,
                    "plan_efficiency": plan_efficiency,
                    "total_tokens": run_log["total_tokens"],
                    "latency_seconds": run_log["latency_seconds"],
                    "run_log": run_log,
                    "ground_truth": task_data
                }
                all_results.append(result)
                
            except Exception as e:
                print(f"  -> ERROR: {str(e)}")

    # Save all results
    with open("final_benchmark_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
        
    print("\n--- Benchmark Complete! Results saved to 'final_benchmark_results.json' ---")

if __name__ == "__main__":
    run_full_benchmark()