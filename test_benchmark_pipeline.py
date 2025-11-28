import pytest
import json
import os
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd

# Import the logic we want to test
# (Ensure your scripts are named run_benchmark.py and evaluation.py)
from run_benchmark import calculate_success, calculate_plan_efficiency, run_full_benchmark
from evaluation import calculate_tool_metrics, run_evaluation

# 1. TEST BENCHMARK RUNNER LOGIC

def test_runner_metric_calculations():
    """Test the helper functions inside run_benchmark.py"""
    
    # 1. Test Success Calculation (Keyword Match)
    log_success = {"final_answer": "The risks are high blood pressure and obesity."}
    ground_truth = {"gold_end_state": {"answer": "obesity"}}
    assert calculate_success(log_success, ground_truth) is True

    log_fail = {"final_answer": "I don't know."}
    assert calculate_success(log_fail, ground_truth) is False

    # 2. Test Plan Efficiency
    # Case: Perfect efficiency (Oracle=2, Agent=2)
    log_eff = {"steps": [{"type": "tool_call"}, {"type": "tool_call"}]}
    gt_eff = {"oracle_plan": [{"tool_name": "a"}, {"tool_name": "b"}]}
    assert calculate_plan_efficiency(log_eff, gt_eff) == 1.0

    # Case: Inefficient (Oracle=1, Agent=2)
    gt_ineff = {"oracle_plan": [{"tool_name": "a"}]}
    assert calculate_plan_efficiency(log_eff, gt_ineff) == 0.5

@patch("run_benchmark.os.listdir")
@patch("run_benchmark.open")
@patch("run_benchmark.json.load")
@patch("run_benchmark.json.dump")
@patch("run_benchmark.NoToolAgent") # Mock the agents class
def test_run_benchmark_flow(mock_NoTool, mock_dump, mock_load, mock_open, mock_listdir):
    """
    Tests the main execution loop of run_benchmark.py without APIs.
    """
    # 1. Setup Mocks
    mock_listdir.return_value = ["task_01.json"]
    
    # Mock the content of task_01.json
    mock_task_data = {
        "task_id": 1,
        "prompt": "Test Prompt",
        "agent_classification": "No-Tool",
        "gold_end_state": {"answer": "test answer"},
        "oracle_plan": []
    }
    mock_load.return_value = mock_task_data
    
    # Mock the Agent's behavior
    mock_agent_instance = mock_NoTool.return_value
    mock_agent_instance.run.return_value = {
        "final_answer": "This is a test answer",
        "steps": [],
        "total_tokens": 100,
        "latency_seconds": 1.5,
        "agent_type": "no_tool"
    }

    # 2. Run the Benchmark Function
    run_full_benchmark()

    # 3. Verify it tried to save results
    assert mock_dump.called
    # Get the data that was passed to json.dump
    saved_data = mock_dump.call_args[0][0]
    
    # Verify the structure of the saved results
    assert len(saved_data) == 2 # 2 models (Gemini & GPT) * 1 task = 2 results
    assert saved_data[0]["model"] == "gemini-2.5-flash-preview-09-2025"
    assert saved_data[0]["success"] is True # "test answer" is in "This is a test answer"
    assert saved_data[1]["model"] == "gpt-4o-mini"


# 2. TEST EVALUATION METRIC LOGIC

def test_tool_metric_calculations():
    """Test the precision/recall logic in evaluation.py"""
    
    # Scenario: Oracle needed [Search, Extract]. Agent did [Search, Search, Extract]
    run_log = {
        "steps": [
            {"type": "tool_call", "tool_name": "web_search"},
            {"type": "tool_call", "tool_name": "web_search"}, # Redundant
            {"type": "tool_call", "tool_name": "content_extractor"}
        ]
    }
    oracle_plan = [
        {"tool_name": "web_search"},
        {"tool_name": "content_extractor"}
    ]

    precision, recall, efficiency = calculate_tool_metrics(run_log, oracle_plan)

    # Precision: 2 correct calls / 3 total calls = 0.66
    assert precision == pytest.approx(2/3)
    
    # Recall: 2 required tools found / 2 required = 1.0
    assert recall == 1.0
    
    # Efficiency: 2 needed / 3 taken = 0.66
    assert efficiency == pytest.approx(2/3)

# 3. TEST EVALUATION REPORT GENERATION

@patch("evaluation.pd.DataFrame.to_markdown") # Don't actually print/save files
@patch("evaluation.open")
@patch("evaluation.json.load")
def test_evaluation_report_flow(mock_load, mock_open, mock_to_markdown):
    """
    Tests that evaluation.py correctly aggregates data into a DataFrame.
    """
    # 1. Create Fake Results Data (Mimicking final_benchmark_results.json)
    fake_results = [
        {
            "task_id": 1,
            "model": "gpt-4o-mini",
            "agent_type": "Multi-Tool",
            "run_log": {
                "final_answer": "Correct answer",
                "total_tokens": 1000,
                "latency_seconds": 2.0,
                "steps": [{"type": "tool_call", "tool_name": "content_extractor"}]
            },
            "ground_truth": {
                "task_type": "Standard",
                "gold_end_state": {"answer": "Correct answer"},
                "oracle_plan": [{"tool_name": "content_extractor"}]
            }
        },
        {
            "task_id": 2,
            "model": "gpt-4o-mini", # Same model, different task
            "agent_type": "Multi-Tool",
            "run_log": {
                "final_answer": "Wrong answer",
                "total_tokens": 500,
                "latency_seconds": 1.0,
                "steps": []
            },
            "ground_truth": {
                "task_type": "Standard",
                "gold_end_state": {"answer": "Right answer"},
                "oracle_plan": [{"tool_name": "web_search"}]
            }
        }
    ]
    
    mock_load.return_value = fake_results

    # 2. Run Evaluation
    # We catch the print output to stop it cluttering the test console
    with patch('builtins.print'): 
        run_evaluation()

    # 3. Verify Aggregation Logic via Mocks
    # We can check if pandas was used to create a markdown table
    assert mock_to_markdown.called