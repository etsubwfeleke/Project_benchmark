import glob
import json
import re
import time
from typing import Dict, Optional, Tuple
import os
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI

# Load credentials from .env file
load_dotenv() 
# OPENAI_API_KEY and GOOGLE_API_KEY should be set in the .env file
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else print("Error: OPENAI_API_KEY not found in .env")

google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    print("Error: GOOGLE_API_KEY not found in .env Gemini models will not be available.")

DEBUG_MODE = True

def call_llm(model_name: str, prompt: str) -> Tuple[str, int]:
    """ 
    Helper function to call the specified LLM and return output and token count.
    
    Args: 
        model_name: e.g., "gpt-4", "gemini-pro"
        prompt: the prompt 
    
    Returns:
        A tuple of (response_text, token_count)
    """
    if DEBUG_MODE:
        print(f"\n[DEBUG] Calling LLM: {model_name} with prompt length {len(prompt)}")
        
    if not prompt:
        return "Error: Prompt cannot be empty.", 0

    try:
        
        if "gemini" in model_name.lower():
            if not google_api_key:
                return "Error: GOOGLE_API_KEY not found in .env", 0
            
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            if not response.parts:
                return f"Error: Response was blocked due to {response.prompt_feedback.block_reason}", 0
            
            text = response.text
            tokens = 0
            if hasattr(response, "usage_metadata"):
                tokens = response.usage_metadata.total_token_count
            
            return text, tokens
        
        elif "gpt" in model_name.lower():
            if not openai_client:
                return "Error: OPENAI_API_KEY not found in .env", 0
            
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens
            return text, tokens
        
        else:
            return f"Error: Unsupported model {model_name}", 0
        
    except Exception as e:
        print(f"API Call Error ({model_name}): {e}")
        return f"API Call Error: ({model_name}): {str(e)}", 0

# STATIC KNOWLEDGE BASE (For Reproducible Web Search)

def load_static_documents(directory: str = "knowledge_base") -> Dict[str, str]:
    """
    Loads all .txt files from the specified directory into a dictionary.
    Key = filename (without extension), Value = file content.
    """
    documents = {}
    
    # Find all .txt files in the directory
    search_path = os.path.join(directory, "*.txt")
    files = glob.glob(search_path)
    
    if not files:
        print(f"Warning: No documents found in '{directory}'. Web search will be empty.")
        return {}

    for file_path in files:
        try:
            # Get filename without extension (e.g., 'diabetes')
            filename = os.path.basename(file_path).replace(".txt", "")
            
            with open(file_path, "r", encoding="utf-8") as f:
                documents[filename] = f.read().strip()
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return documents


STATIC_DOCUMENTS = load_static_documents()

STOP_WORDS = set(["a", "is", "in", "what", "the", "for", "and", "of", "to", "was", "it", "with", "as"])

def web_search(query: str) -> str:
    """
    Static web search tool that searches over local documents.
    This ensures reproducibility - no live API calls.
    """
    query_lower = query.lower()
    query_words = [word.strip("?.,") for word in query_lower.split() if word not in STOP_WORDS]
    
    results = []
    for doc_key, doc_content in STATIC_DOCUMENTS.items():
        doc_key_as_search_term = doc_key.replace("_", " ")
        
        # 1. Exact key match in query
        if doc_key_as_search_term in query_lower:
            results.append(f"[Document: {doc_key}]\n{doc_content.strip()}")
            continue 
            
        # 2. Key word match
        key_word_match = False
        for word in query_words:
            if word in doc_key: # e.g., 'heart' in 'heart_disease'
                key_word_match = True
                break
        
        if key_word_match:
            results.append(f"[Document: {doc_key}]\n{doc_content.strip()}")
            continue

        # 3. Content match 
        content_match = False
        for word in query_words:
            if word and word in doc_content.lower(): # 'heart' in '...heart disease...'
                content_match = True
                break
        
        if content_match:
            results.append(f"[Document: {doc_key}]\n{doc_content.strip()}")
            
    if results:
        return "\n\n".join(list(dict.fromkeys(results)))
    else:
        return "No relevant documents found for your query."

def content_extractor(text: str, model_name: str = "gpt-4o-mini") -> Tuple[str, int]:
    """
    Extracts and summarizes key information from text using an LLM.
    
    Args:  
        text: The text to extract key information from  
        model_name: The LLM model to use (default: "gpt-4o-mini")  

    Returns:  
        Tuple[str, int]: A tuple of (extracted_summary, token_count)  
    """
    prompt = f"""You are a content extraction specialist. Extract the key information from the following text and provide a concise summary.

Text to extract from:
{text}

Provide a clear, structured summary:"""
    return call_llm(model_name, prompt)


def quiz_generator(text: str, model_name: str = "gpt-4o-mini") -> Tuple[str, int]:
    """
    Generates a multiple-choice quiz from provided text using an LLM.
    
    Args:
        text: The text to generate quiz questions from
        model_name: The LLM model to use (default: "gpt-4o-mini")

    Returns:
        Tuple[str, int]: A tuple of (quiz_text, token_count)
    """
    
    prompt = f"""You are a quiz generator. Create a 3-question multiple-choice quiz based on the following text. Each question should have 4 options (A, B, C, D) with one correct answer.

Text:
{text}

Generate the quiz in this format:
Question 1: [question text]
A) [option]
B) [option]
C) [option]
D) [option]
Correct Answer: [letter]"""
    
    return call_llm(model_name, prompt)

class NoToolAgent:
    """
    Simplest agent - directly queries LLM without any tools.
    Returns a dictionary log of the run.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
    
    def run(self, prompt: str) -> Dict:
        """Execute the agent and return a log of the run."""
        
        run_log = {
            "agent_type": "no_tool",
            "start_time": time.time(),
            "steps": [],
            "total_tokens": 0,
            "final_answer": None,
            "latency_seconds": 0
        }
        
        final_answer, tokens = call_llm(self.model_name, prompt)
        
        run_log["steps"].append({
            "type": "llm_call",
            "model": self.model_name,
            "input": prompt,
            "output": final_answer,
            "tokens": tokens
        })
        run_log["total_tokens"] = tokens
        run_log["final_answer"] = final_answer
        run_log["end_time"] = time.time()
        run_log["latency_seconds"] = run_log["end_time"] - run_log["start_time"]
        
        return run_log


class SingleToolAgent:
    """
    ReAct-style agent with web_search tool.
    Returns a dictionary log of the run.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.tool_definition = {
            "name": "web_search",
            "description": "Search for information in a knowledge base. Use this when you need specific factual information.",
            "parameters": {"query": "The search query string"}
        }
    
    def _parse_tool_call(self, llm_response: str) -> Optional[Dict]:
        action_match = re.search(r'Action:\s*(\w+)', llm_response)
        input_match = re.search(r'Action Input:\s*[\"\']?(.*?)[\"\']?\s*(?:\nObservation:|\n|Thought:|$)', llm_response, re.DOTALL)
        
        if action_match and input_match:
            return {"tool": action_match.group(1), "query": input_match.group(1).strip()}
        return None
    
    def run(self, prompt: str) -> Dict:
        """
        Execute the agent and return a log of the run.
        """
        run_log = {
            "agent_type": "single_tool",
            "start_time": time.time(),
            "steps": [],
            "total_tokens": 0,
            "final_answer": None,
            "latency_seconds": 0
        }

        react_prompt = f"""You are a helpful assistant with access to tools. Use the following format:

Question: the input question
Thought: think about what to do
Action: the action to take (must be: web_search)
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the question

Available Tools:
- web_search: {self.tool_definition['description']}

Question: {prompt}
Thought:"""
        llm_response_1, tokens_1 = call_llm(self.model_name, react_prompt)
        
        run_log["steps"].append({
            "type": "llm_call",
            "model": self.model_name,
            "input": react_prompt,
            "output": llm_response_1,
            "tokens": tokens_1
        })
        run_log["total_tokens"] += tokens_1
        
        tool_call = self._parse_tool_call(llm_response_1)
        
        if tool_call:
            if tool_call["tool"] == "web_search":
                search_result = web_search(tool_call["query"])
                
                run_log["steps"].append({
                    "type": "tool_call",
                    "tool_name": "web_search",
                    "input": tool_call["query"],
                    "output": search_result
                })
                
                final_prompt = f"{react_prompt}\n{llm_response_1}\nObservation: {search_result}\nThought:"
                llm_response_2, tokens_2 = call_llm(self.model_name, final_prompt)
                
                if "Final Answer:" in llm_response_2:
                    final_answer = llm_response_2.split("Final Answer:")[-1].strip()
                else:
                    final_answer = llm_response_2

                run_log["steps"].append({
                    "type": "llm_call",
                    "model": self.model_name,
                    "input": final_prompt,
                    "output": llm_response_2,
                    "tokens": tokens_2
                })
                run_log["total_tokens"] += tokens_2
                run_log["final_answer"] = final_answer
            else:
                run_log["final_answer"] = f"Error: Tool '{tool_call['tool']}' is not available. Only 'web_search' is supported." 
                
        else:
            # If agent didn't call tool, just return what it said
            run_log["final_answer"] = llm_response_1
        
        run_log["end_time"] = time.time()
        run_log["latency_seconds"] = run_log["end_time"] - run_log["start_time"]
        return run_log


class MultiToolAgent:
    """
    ReAct-style agent with multiple tools.
    Returns a dictionary log of the run.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.tool_definitions = {
            "content_extractor": {"name": "content_extractor", "description": "Extract and summarize key information from a text document."},
            "quiz_generator": {"name": "quiz_generator", "description": "Generate a multiple-choice quiz from provided text."}
        }
    
    def _parse_tool_call(self, llm_response: str) -> Optional[Dict]:
        action_match = re.search(r'Action:\s*(\w+)', llm_response)
        input_match = re.search(r'Action Input:\s*["\']?(.+?)["\']?(?:\n|$)', llm_response, re.DOTALL)
        if action_match and input_match:
            return {"tool": action_match.group(1), "input": input_match.group(1).strip()}
        return None

    def run(self, prompt: str) -> Dict:
        """
        Execute the agent and return a log of the run.
        """
        run_log = {
            "agent_type": "multi_tool",
            "start_time": time.time(),
            "steps": [],
            "total_tokens": 0,
            "final_answer": None,
            "latency_seconds": 0
        }
        
        tools_desc = "\n".join([f"- {name}: {tool['description']}" for name, tool in self.tool_definitions.items()])
        react_prompt = f"""You are a helpful assistant with access to multiple tools. Use the following format:

Question: the input question
Thought: think about what to do
Action: the action to take (must be one of: content_extractor, quiz_generator)
Action Input: the input to the action
Observation: the result of the action
...
Final Answer: the final answer to the question

Available Tools:
{tools_desc}

Question: {prompt}
Thought:"""
        llm_response_1, tokens_1 = call_llm(self.model_name, react_prompt)
        
        run_log["steps"].append({
            "type": "llm_call",
            "model": self.model_name,
            "input": react_prompt,
            "output": llm_response_1,
            "tokens": tokens_1
        })
        run_log["total_tokens"] += tokens_1
        
        tool_call = self._parse_tool_call(llm_response_1)
        
        if tool_call:
            tool_name = tool_call["tool"]
            tool_input = tool_call["input"]
            tool_result = ""
            tool_token_cost = 0
            
            if tool_name == "content_extractor":
                tool_result, tool_token_cost = content_extractor(tool_input, model_name=self.model_name)
            elif tool_name == "quiz_generator":
                tool_result, tool_token_cost = quiz_generator(tool_input, model_name=self.model_name)
            else:
                tool_result = f"Error: Unknown tool {tool_name}"
                tool_token_cost = 0

            run_log["steps"].append({
                "type": "tool_call",
                "tool_name": tool_name, 
                "input": tool_input,
                "output": tool_result,
                "tokens": tool_token_cost
            })
            run_log["total_tokens"] += tool_token_cost
            
            final_prompt = f"{react_prompt}\n{llm_response_1}\nObservation: {tool_result}\nThought:"
            llm_response_2, tokens_2 = call_llm(self.model_name, final_prompt)
            
            if "Final Answer:" in llm_response_2:
                final_answer = llm_response_2.split("Final Answer:")[-1].strip()
            else:
                final_answer = llm_response_2
                
            run_log["steps"].append({
                "type": "llm_call",
                "model": self.model_name,
                "input": final_prompt,
                "output": llm_response_2,
                "tokens": tokens_2
            })
            run_log["total_tokens"] += tokens_2
            run_log["final_answer"] = final_answer
        else:
            run_log["final_answer"] = llm_response_1
        run_log["end_time"] = time.time()
        run_log["latency_seconds"] = run_log["end_time"] - run_log["start_time"]
        return run_log

def main():
    print("="*60)
    print("TESTING LIVE AGENTS (OpenAI & Gemini)")
    print("="*60)

    # 1. Test OpenAI
    print("\n--- Testing SingleToolAgent with GPT-4o-mini ---")
    agent_gpt = SingleToolAgent("gpt-4o-mini")
    log_gpt = agent_gpt.run("What are the risk factors for heart disease?")
    final_answer = log_gpt["final_answer"] or ""
    print(f"Final Answer: {final_answer[:150]}...")
    print(f"Total Tokens: {log_gpt['total_tokens']}")

    # 2. Test Gemini
    print("\n--- Testing SingleToolAgent with Gemini-1.5-flash ---")
    agent_gemini = SingleToolAgent("gemini-1.5-flash")
    log_gemini = agent_gemini.run("What are the risk factors for heart disease?")
    final_answer = log_gemini["final_answer"] or ""
    print(f"Final Answer: {final_answer[:150]}...")
    print(f"Total Tokens: {log_gemini['total_tokens']}")

if __name__ == "__main__":
    main()