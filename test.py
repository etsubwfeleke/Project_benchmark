import pytest
from unittest.mock import patch
from agents import (
    NoToolAgent, 
    SingleToolAgent, 
    MultiToolAgent,
    web_search
)

def test_web_search_tool_success():
    """Tests that the static web_search tool finds the correct document."""
    query = "information on diabetes"
    result = web_search(query)
    assert "Diabetes is a chronic health condition" in result
    assert "Document: diabetes" in result

@patch("agents.call_llm")
def test_no_tool_agent_log_structure(mock_call_llm):
    """
    Tests NoToolAgent by simulating a successful LLM response.
    """
    # 1. Define what the "LLM" should return
    mock_call_llm.return_value = ("Based on my internal knowledge, diabetes is...", 50)
    
    agent = NoToolAgent()
    prompt = "What is diabetes?"
    log = agent.run(prompt)
    
    # 2. Verify the log structure
    assert log["agent_type"] == "no_tool"
    assert "Based on my internal knowledge" in log["final_answer"]
    assert len(log["steps"]) == 1
    assert log["total_tokens"] == 50

@patch("agents.call_llm")
def test_single_tool_agent_search_integration_and_log(mock_call_llm):
    """
    Tests SingleToolAgent logic by mocking the 2-step ReAct conversation.
    """
    # 1. Setup the sequence of LLM responses
    # Call 1: The agent decides to search
    response_1 = ("I need to search.\nAction: web_search\nAction Input: heart disease", 70)
    # Call 2: The agent gives the final answer
    response_2 = ("Based on the search results, Heart disease refers to...", 30)
    
    mock_call_llm.side_effect = [response_1, response_2]
    
    agent = SingleToolAgent()
    log = agent.run("What is heart disease?")
    
    # 2. Verify logic
    # Check if web_search was actually called in the log
    tool_step = log["steps"][1]
    assert tool_step["type"] == "tool_call"
    assert tool_step["tool_name"] == "web_search"
    assert "Heart disease refers to" in tool_step["output"] # Logic check: did tool run?
    
    assert "Based on the search results" in log["final_answer"]
    assert log["total_tokens"] == 100

@patch("agents.call_llm")
def test_multi_tool_quiz_generator_route_and_log(mock_call_llm):
    """
    Tests MultiToolAgent routing to Quiz Generator.
    """
    # Call 1: Route to quiz tool
    response_1 = ("Action: quiz_generator\nAction Input: nutrition text", 80)
    # Call 2: Final Answer
    response_2 = ("Final Answer: Question 1: What is a key prevention strategy...", 25)
    
    # We also need to mock content_extractor/quiz_generator internal calls
    # But since your code calls 'call_llm' inside the tool, we just add it to the side_effect list!
    # Order: 1. Agent Reasoning -> 2. Tool Execution (calls LLM) -> 3. Agent Final Answer
    
    tool_execution_response = ("Question 1: ...", 200)
    
    mock_call_llm.side_effect = [response_1, tool_execution_response, response_2]
    
    agent = MultiToolAgent()
    log = agent.run("Generate a quiz about nutrition")
    
    # Check that the correct tool was logged
    tool_step = log["steps"][1]
    assert tool_step["tool_name"] == "quiz_generator"
    assert "Question 1:" in log["final_answer"]

@patch("agents.call_llm")
def test_multi_tool_content_extractor_route_and_log(mock_call_llm):
    """
    Tests MultiToolAgent routing to Content Extractor.
    """
    # Order: 1. Agent Reasoning -> 2. Tool Execution (calls LLM) -> 3. Agent Final Answer
    response_1 = ("Action: content_extractor\nAction Input: exercise text", 80)
    tool_response = ("[EXTRACTED CONTENT]...", 150)
    response_2 = ("Final Answer: [EXTRACTED CONTENT]...", 25)
    
    mock_call_llm.side_effect = [response_1, tool_response, response_2]
    
    agent = MultiToolAgent()
    log = agent.run("Summarize this text about exercise")
    
    # Check logic
    assert log["steps"][1]["tool_name"] == "content_extractor"
    assert "[EXTRACTED CONTENT]" in log["final_answer"]