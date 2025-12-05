import json
import pandas as pd
import os

def run_paired_analysis(results_file="final_benchmark_results_trial_2.json"):
    if not os.path.exists(results_file):
        print(f"Error: File '{results_file}' not found.")
        return

    print(f"Loading results from {results_file}...")
    with open(results_file, 'r') as f:
        data = json.load(f)

    # 1. Organize Data by Task ID and Agent Type
    # Structure: { (task_id, agent_type): { model_name: result_entry } }
    tasks_map = {}

    for entry in data:
        task_id = entry.get('task_id')
        agent_type = entry.get('agent_type')
        model = entry.get('model')
        
        key = (task_id, agent_type)
        
        if key not in tasks_map:
            tasks_map[key] = {}
        
        tasks_map[key][model] = entry

    # 2. Perform Paired Comparison
    # We want to compare the two models for each (task_id, agent_type) pair.
    # Assuming the two models are:
    # model_a = "gemini-2.5-flash-preview-09-2025"
    # model_b = "gpt-4o-mini"
    
    # Auto-detect models if possible, or hardcode for consistency
    models_found = set()
    for entry in data:
        models_found.add(entry.get('model'))
    
    models_list = sorted(list(models_found))
    if len(models_list) < 2:
        print("Error: Need at least 2 models to compare.")
        print(f"Found models: {models_list}")
        return
        
    model_a = models_list[0] # e.g. gemini
    model_b = models_list[1] # e.g. gpt-4o-mini

    print(f"Comparing Model A: {model_a}")
    print(f"          Model B: {model_b}")
    print("-" * 60)

    comparison_results = []

    for (task_id, agent_type), model_results in tasks_map.items():
        res_a = model_results.get(model_a)
        res_b = model_results.get(model_b)

        if not res_a or not res_b:
            continue # Skip if one model didn't run this task

        success_a = res_a.get('success', False)
        success_b = res_b.get('success', False)

        status = "Both Failed"
        if success_a and success_b:
            status = "Both Passed"
        elif success_a and not success_b:
            status = f"{model_a} Passed, {model_b} Failed"
        elif not success_a and success_b:
            status = f"{model_b} Passed, {model_a} Failed"

        comparison_results.append({
            "Task ID": task_id,
            "Agent Type": agent_type,
            "Status": status,
            f"{model_a} Success": success_a,
            f"{model_b} Success": success_b,
             f"{model_a} Tool": _get_tool_usage(res_a),
            f"{model_b} Tool": _get_tool_usage(res_b)
        })

    # 3. Create DataFrame and Summary
    df = pd.DataFrame(comparison_results)
    
    # Summary Table
    summary = df.groupby(['Agent Type', 'Status']).size().unstack(fill_value=0)
    
    print("\n=== PAIRED ANALYSIS SUMMARY ===")
    print(summary.to_markdown())

    # Save detailed comparison
    output_file = "paired_analysis_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed paired analysis saved to '{output_file}'")

def _get_tool_usage(entry):
    """Helper to extract tool names used in a run"""
    if not entry or 'run_log' not in entry or 'steps' not in entry['run_log']:
        return "N/A"
    
    tools = [s['tool_name'] for s in entry['run_log']['steps'] if s['type'] == 'tool_call']
    return ", ".join(tools) if tools else "None"

if __name__ == "__main__":
    run_paired_analysis()