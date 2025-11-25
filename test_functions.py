import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
from agents import call_llm, load_static_documents

@patch("agents.openai_client")
def test_call_llm_openai_success(mock_openai):
    """Test successful call to OpenAI GPT model."""
    # 1. Setup the mock response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is a GPT response."
    mock_response.usage.total_tokens = 42
    mock_openai.chat.completions.create.return_value = mock_response

    # 2. Run function
    response, tokens = call_llm("gpt-4o-mini", "Hello")

    # 3. Verify results
    assert response == "This is a GPT response."
    assert tokens == 42
    mock_openai.chat.completions.create.assert_called_once()

@patch("agents.genai.GenerativeModel")
def test_call_llm_gemini_success(mock_genai_model):
    """Test successful call to Google Gemini model."""
    # 1. Setup the mock model and response
    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is a Gemini response."
    mock_response.parts = [True] # Simulate safety check pass
    # Mock the usage_metadata attribute
    mock_response.usage_metadata.total_token_count = 15
    
    mock_genai_model.return_value = mock_model_instance
    mock_model_instance.generate_content.return_value = mock_response

    # 2. Run function
    # We need to mock the GOOGLE_API_KEY check too
    with patch("agents.google_api_key", "fake_key"):
        response, tokens = call_llm("gemini-1.5-flash", "Hello")

    # 3. Verify results
    assert response == "This is a Gemini response."
    assert tokens == 15

def test_call_llm_empty_prompt():
    """Test handling of empty prompt."""
    response, tokens = call_llm("gpt-4o-mini", "")
    assert response == ""
    assert tokens == 0

def test_call_llm_unsupported_model():
    """Test handling of unknown model name."""
    response, tokens = call_llm("llama-3-70b", "Hello")
    assert "Error: Unsupported model" in response
    assert tokens == 0

@patch("agents.openai_client")
def test_call_llm_api_exception(mock_openai):
    """Test handling of API errors/exceptions."""
    # Force the mock to raise an exception
    mock_openai.chat.completions.create.side_effect = Exception("Rate limit exceeded")

    response, tokens = call_llm("gpt-4o-mini", "Hello")
    
    assert "Error calling API" in response
    assert "Rate limit exceeded" in response
    assert tokens == 0

def test_call_llm_missing_api_key():
    """Test graceful failure when keys are None."""
    # We patch the client object in agents.py to be None
    with patch("agents.openai_client", None):
        response, tokens = call_llm("gpt-4o-mini", "Hello")
        assert "Error: OPENAI_API_KEY not found" in response
        
        
@patch("agents.glob.glob")
@patch("builtins.open", new_callable=mock_open, read_data="Fake content")
def test_load_documents_success(mock_file, mock_glob):
    """Test successfully loading multiple .txt files."""
    # Simulate finding two files
    mock_glob.return_value = ["knowledge_base/doc1.txt", "knowledge_base/doc2.txt"]

    docs = load_static_documents("knowledge_base")

    assert len(docs) == 2
    assert "doc1" in docs
    assert "doc2" in docs
    assert docs["doc1"] == "Fake content"

@patch("agents.glob.glob")
def test_load_documents_empty_directory(mock_glob):
    """Test behavior when directory is empty or not found."""
    mock_glob.return_value = [] # No files found

    docs = load_static_documents("missing_folder")
    
    assert docs == {}
    assert isinstance(docs, dict)

@patch("agents.glob.glob")
def test_load_documents_read_error(mock_glob):
    """Test handling of a corrupted file among good ones."""
    mock_glob.return_value = ["good.txt", "bad.txt"]

    # Define a side_effect for open(): 
    # First call returns a valid file, second raises IOError
    
    # We need a complex mock for open because it's a context manager
    valid_file_mock = mock_open(read_data="Good content").return_value
    
    def side_effect(filename, *args, **kwargs):
        if "bad.txt" in filename:
            raise IOError("Disk error")
        return valid_file_mock

    with patch("builtins.open", side_effect=side_effect):
        docs = load_static_documents("knowledge_base")

    # Should load the good one and skip the bad one without crashing
    assert len(docs) == 1
    assert "good" in docs
    assert "bad" not in docs