import pytest
from agents import (
    NoToolAgent, 
    SingleToolAgent, 
    MultiToolAgent,
    web_search,
    STATIC_DOCUMENTS
)

def test_web_search_tool_success():
    """Tests that the static web_search tool finds the correct document."""
    query = "information on diabetes"
    result = web_search(query)
    assert "Diabetes is a chronic health condition" in result
    assert "Document: diabetes" in result

def test_no_tool_agent_log_structure():
    """
    Tests the NoToolAgent's stubbed response.
    This confirms the agent's run method executes and returns a valid log.
    """
    agent = NoToolAgent()
    prompt = "What is diabetes?"
    log = agent.run(prompt)
    
    assert log["agent_type"] == "no_tool"
    assert "Based on my internal knowledge" in log["final_answer"]
    assert len(log["steps"]) == 1
    assert log["steps"][0]["type"] == "llm_call"
    assert log["total_tokens"] > 0
    assert log["latency_seconds"] >= 0

def test_single_tool_agent_search_integration_and_log():
    """
    Tests the SingleToolAgent's full stubbed logic.
    This checks if the agent:
    1. Simulates a tool call (Action: web_search)
    2. Correctly calls the *actual* web_search function.
    3. Integrates the web_search result into its final simulated answer.
    4. Logs all steps correctly.
    """
    agent = SingleToolAgent()
    prompt = "What is heart disease?"
    log = agent.run(prompt)
    
    # Check final answer
    assert "Based on the search results" in log["final_answer"]
    assert "Heart disease refers to" in log["final_answer"]
    
    # Check log structure
    assert log["agent_type"] == "single_tool"
    assert len(log["steps"]) == 3 # llm_call -> tool_call -> llm_call
    assert log["steps"][0]["type"] == "llm_call"
    assert log["steps"][1]["type"] == "tool_call"
    assert log["steps"][1]["tool_name"] == "web_search"
    assert log["steps"][1]["output"] == web_search(prompt) # Confirms tool was called
    assert log["steps"][2]["type"] == "llm_call"
    assert log["total_tokens"] > 0

def test_multi_tool_quiz_generator_route_and_log():
    """
    Tests if the MultiToolAgent correctly routes to the 
    quiz_generator tool and logs it.
    """
    agent = MultiToolAgent()
    prompt = "Generate a quiz about nutrition"
    log = agent.run(prompt)
    
    # Check final answer
    assert "Question 1: What is a key prevention strategy" in log["final_answer"]
    
    # Check log structure
    assert log["agent_type"] == "multi_tool"
    assert len(log["steps"]) == 3 # llm_call -> tool_call -> llm_call
    assert log["steps"][1]["type"] == "tool_call"
    assert log["steps"][1]["tool_name"] == "quiz_generator"

def test_multi_tool_content_extractor_route_and_log():
    """
    Tests if the MultiToolAgent correctly routes to the 
    content_extractor tool and logs it.
    """
    agent = MultiToolAgent()
    prompt = "Summarize this text about exercise"
    log = agent.run(prompt)
    
    # Check final answer
    assert "[EXTRACTED CONTENT]" in log["final_answer"]
    
    # Check log structure
    assert log["steps"][1]["type"] == "tool_call"
    assert log["steps"][1]["tool_name"] == "content_extractor"

