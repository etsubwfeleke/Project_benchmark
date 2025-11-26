import json
import re
import time
import glob
from typing import Dict, List, Optional, Tuple, Union
import os
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI

# Load credentials from .env file
load_dotenv()
# Configure OpenAI Client
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# Configure Google Gemini Client
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)

def call_llm(model_name: str, prompt: str) -> Tuple[str, int]:
    """
    Unified helper function to call either OpenAI or Google Gemini APIs.
    Handles model-specific logic and token counting.

    Args:
        model_name (str): The model identifier (e.g., "gpt-4o-mini", "gemini-1.5-flash").
        prompt (str): The text prompt to send to the model.

    Returns:
        Tuple[str, int]: A tuple containing (response_text, total_token_count).
        Returns an error message and 0 tokens on failure.
    """
    if not prompt:
        return "", 0

    try:
        # --- GOOGLE GEMINI LOGIC ---
        if "gemini" in model_name.lower():
            if not google_api_key:
                return "Error: GOOGLE_API_KEY not found in .env", 0
                
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            # Check for safety blocks or empty responses
            if not response.parts:
                return "Error: Model blocked response (Safety).", 0
                
            text = response.text
            # Access token usage metadata
            tokens = 0
            if hasattr(response, "usage_metadata"):
                tokens = response.usage_metadata.total_token_count
            
            return text, tokens

        # --- OPENAI GPT LOGIC ---
        elif "gpt" in model_name.lower():
            if not openai_client:
                return "Error: OPENAI_API_KEY not found in .env", 0

            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0 # Use deterministic setting for consistent benchmarks
            )
            
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens
            return text, tokens
        
        else:
            return f"Error: Unsupported model '{model_name}'", 0

    except Exception as e:
        print(f"API Call Error ({model_name}): {e}")
        return f"Error calling API: {str(e)}", 0

def load_static_documents(directory: str = "knowledge_base") -> Dict[str, str]:
    """
    Loads all .txt files from the specified directory into a dictionary.
    This simulates a 'database' for the search tool, ensuring reproducibility.

    Args:
        directory (str): The folder name containing text documents.

    Returns:
        Dict[str, str]: Dictionary where key is filename (no extension) and value is content.
    """
    documents = {}
    # Construct path relative to this script file
    base_path = os.path.dirname(os.path.abspath(__file__))
    search_path = os.path.join(base_path, directory, "*.txt")
    
    files = glob.glob(search_path)
    
    if not files:
        print(f"Warning: No documents found in '{search_path}'. Web search will be empty.")
        return {}

    print(f"Loading Knowledge Base from: {directory}")
    for file_path in files:
        try:
            filename = os.path.basename(file_path).replace(".txt", "")
            with open(file_path, "r", encoding="utf-8") as f:
                documents[filename] = f.read().strip()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    print(f"Loaded {len(documents)} documents.")
    return documents

# Initialize the knowledge base once at startup
STATIC_DOCUMENTS = load_static_documents()

# Common stop words to filter out during naive search logic
STOP_WORDS = set(["a", "is", "in", "what", "the", "for", "and", "of", "to", "was", "it", "with", "as"])

def web_search(query: str) -> str:
    """
    Simulates a web search engine by querying the local static document set.
    This ensures that search results are deterministic and reproducible across runs.

    Args:
        query (str): The search query string.

    Returns:
        str: The content of relevant documents or a "No results" message.
    """
    query_lower = query.lower()
    # Filter out stop words for better keyword matching
    query_words = [word.strip("?.,") for word in query_lower.split() if word not in STOP_WORDS]
    
    results = []
    for doc_key, doc_content in STATIC_DOCUMENTS.items():
        doc_key_as_search_term = doc_key.replace("_", " ")
        
        # Priority 1: Exact key match in query (e.g. query contains "diabetes")
        if doc_key_as_search_term in query_lower:
            results.append(f"[Document: {doc_key}]\n{doc_content.strip()}")
            continue 
            
        # Priority 2: Key word match in document title
        key_word_match = False
        for word in query_words:
            if word in doc_key:
                key_word_match = True
                break
        
        if key_word_match:
            results.append(f"[Document: {doc_key}]\n{doc_content.strip()}")
            continue

        # Priority 3: Content match (word appears in document body)
        content_match = False
        for word in query_words:
            if word and word in doc_content.lower():
                content_match = True
                break
        
        if content_match:
            results.append(f"[Document: {doc_key}]\n{doc_content.strip()}")
            
    if results:
        # Remove duplicates and join
        return "\n\n".join(list(dict.fromkeys(results)))
    else:
        return "No relevant documents found for your query."


def content_extractor(text: str, model_name: str = "gpt-4o-mini") -> Tuple[str, int]:
    """
    Tool: Extracts and summarizes key information from a text.
    Uses the live LLM to perform the summarization.
    """
    prompt = f"""You are an expert content extraction specialist. 
Your task is to analyze the following text, extract the most critical information, and provide a concise, structured summary.
Focus on key facts, dates, and definitions.

Text to process:
{text}

Output a clear summary:"""
    
    return call_llm(model_name, prompt)


def quiz_generator(text: str, model_name: str = "gpt-4o-mini") -> Tuple[str, int]:
    """
    Tool: Generates a multiple-choice quiz based on the provided text.
    Uses the live LLM to generate questions.
    """
    prompt = f"""You are an educational quiz generator. 
Your task is to create a 3-question multiple-choice quiz based STRICTLY on the text provided below.
Each question must have 4 options (A, B, C, D) and clearly indicate the correct answer.

Text to process:
{text}

Format your output exactly like this:
Question 1: [Question text]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Correct Answer: [Letter]

(Repeat for Question 2 and 3)"""
    
    return call_llm(model_name, prompt)

class NoToolAgent:
    """
    A baseline agent that answers queries directly using the LLM's internal knowledge.
    It has no access to external tools.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
    
    def run(self, prompt: str) -> Dict:
        """
        Executes the agent logic.
        
        Args:
            prompt (str): The user's query.
            
        Returns:
            Dict: A log of the run, including steps, tokens, and final answer.
        """
        run_log = {
            "agent_type": "no_tool",
            "start_time": time.time(),
            "steps": [],
            "total_tokens": 0,
            "final_answer": None,
            "latency_seconds": 0
        }
        
        # Enhanced prompt for No-Tool agent to ensure it tries its best without tools
        system_prompt = f"""You are a knowledgeable AI assistant. 
You do not have access to external tools like search or databases.
Answer the user's question to the best of your ability using your internal training data.
If you do not know the answer, please state that clearly.

User Question: {prompt}"""

        final_answer, tokens = call_llm(self.model_name, system_prompt)
        
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
    An agent equipped with a single tool: 'web_search'.
    It uses a ReAct (Reason+Act) loop to decide when to search and how to answer.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.tool_definition = {
            "name": "web_search",
            "description": "Search for information in a knowledge base. Use this when you need specific factual information.",
            "parameters": {"query": "The search query string"}
        }
    
    def _parse_tool_call(self, llm_response: str) -> Optional[Dict]:
        """Parses the LLM response to extract Action and Action Input."""
        action_match = re.search(r'Action:\s*(\w+)', llm_response)
        input_match = re.search(r'Action Input:\s*["\']?(.+?)["\']?(?:\n|$)', llm_response, re.DOTALL)
        
        if action_match and input_match:
            return {"tool": action_match.group(1), "query": input_match.group(1).strip()}
        return None

    def _extract_final_answer(self, llm_response: str) -> str:
        """Helper to extract the text following 'Final Answer:'."""
        if "Final Answer:" in llm_response:
            return llm_response.split("Final Answer:")[-1].strip()
        return llm_response
    
    def run(self, prompt: str) -> Dict:
        run_log = {
            "agent_type": "single_tool",
            "start_time": time.time(),
            "steps": [],
            "total_tokens": 0,
            "final_answer": None,
            "latency_seconds": 0
        }

        # Detailed ReAct Prompt to guide the Single-Tool Agent
        react_prompt = f"""You are a helpful assistant with access to a knowledge base via a search tool.
Your goal is to answer the user's question accurately.
ALWAYS use the search tool if the question requires factual information.

Use the following ReAct format:

Question: the input question
Thought: I should think about what information I need.
Action: web_search
Action Input: [the search query]
Observation: [the result of the search will appear here]
... (You can repeat Thought/Action/Observation if needed, but usually once is enough)
Thought: I now have the information to answer.
Final Answer: [your final response to the user]

Available Tools:
- web_search: {self.tool_definition['description']}

Question: {prompt}
Thought:"""

        # 1. Reasoning Step
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
        
        if tool_call and tool_call["tool"] == "web_search":
            # --- INPUT VALIDATION ---
            query = tool_call.get("query", "")
            if not isinstance(query, str):
                query = "" if query is None else str(query)
            query = query.strip()
            
            if not query:
                error_msg = "Error: Search query cannot be empty."
                run_log["final_answer"] = error_msg
                run_log["end_time"] = time.time()
                run_log["latency_seconds"] = run_log["end_time"] - run_log["start_time"]
                return run_log
            # ------------------------

            # Execute Tool
            search_result = web_search(query)
            
            run_log["steps"].append({
                "type": "tool_call",
                "tool_name": "web_search",
                "input": query,
                "output": search_result
            })
            
            # 2. Final Answer Step
            final_prompt = f"{react_prompt}\n{llm_response_1}\nObservation: {search_result}\nThought:"
            llm_response_2, tokens_2 = call_llm(self.model_name, final_prompt)
            
            final_answer = self._extract_final_answer(llm_response_2)

            run_log["steps"].append({
                "type": "llm_call",
                "model": self.model_name,
                "input": final_prompt,
                "output": final_answer,
                "tokens": tokens_2
            })
            run_log["total_tokens"] += tokens_2
            run_log["final_answer"] = final_answer
            
        else:
            # Fallback: If agent refused to use tool or halluncinated
            run_log["final_answer"] = self._extract_final_answer(llm_response_1)
        
        run_log["end_time"] = time.time()
        run_log["latency_seconds"] = run_log["end_time"] - run_log["start_time"]
        return run_log


class MultiToolAgent:
    """
    An agent equipped with multiple specialized tools:
    1. content_extractor (for summaries)
    2. quiz_generator (for creating quizzes)
    
    It must correctly ROUTE the user's request to the appropriate tool.
    """
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.tool_definitions = {
            "content_extractor": {"name": "content_extractor", "description": "Extract and summarize key information from a text document."},
            "quiz_generator": {"name": "quiz_generator", "description": "Generate a multiple-choice quiz from provided text."}
        }
    
    def _parse_tool_call(self, llm_response: str) -> Optional[Dict]:
        """Parses the LLM response for Action and Action Input."""
        action_match = re.search(r'Action:\s*(\w+)', llm_response)
        input_match = re.search(r'Action Input:\s*["\']?(.+?)["\']?(?:\n|$)', llm_response, re.DOTALL)
        if action_match and input_match:
            return {"tool": action_match.group(1), "input": input_match.group(1).strip()}
        return None

    def _extract_final_answer(self, llm_response: str) -> str:
        """Helper to clean up response if model explicitly says 'Final Answer:'"""
        if "Final Answer:" in llm_response:
            return llm_response.split("Final Answer:")[-1].strip()
        return llm_response

    def run(self, prompt: str) -> Dict:
        run_log = {
            "agent_type": "multi_tool",
            "start_time": time.time(),
            "steps": [],
            "total_tokens": 0,
            "final_answer": None,
            "latency_seconds": 0
        }
        
        tools_desc = "\n".join([f"- {name}: {tool['description']}" for name, tool in self.tool_definitions.items()])
        
        # Detailed ReAct Prompt for Routing
        react_prompt = f"""You are a smart assistant with access to specialized tools.
Your main job is to ROUTE the user's request to the correct tool.

GUIDELINES:
1. If the user wants a summary, extraction, or key points, use 'content_extractor'.
2. If the user wants a quiz, test questions, or an exam, use 'quiz_generator'.
3. Do not answer directly if a tool is appropriate.

Format:
Question: [Input]
Thought: [Reasoning about which tool to pick]
Action: [Tool Name]
Action Input: [The text to process]
Observation: [Tool output]
Final Answer: [Final output]

Available Tools:
{tools_desc}

Question: {prompt}
Thought:"""

        # 1. Reasoning Step (Routing)
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
            
            # --- SINGLE DECISION BLOCK FOR EXECUTION ---
            tool_result = ""
            tool_token_cost = 0

            # 1. Validation Check
            if not tool_input: 
                tool_result = "Error: Tool input cannot be empty."
                
            # 2. Route to Content Extractor
            elif tool_name == "content_extractor":
                # Pass model_name to ensure the tool uses the same brain as the agent
                tool_result, tool_token_cost = content_extractor(tool_input, model_name=self.model_name)
                
            # 3. Route to Quiz Generator
            elif tool_name == "quiz_generator":
                tool_result, tool_token_cost = quiz_generator(tool_input, model_name=self.model_name)
                
            # 4. Handle Unknown Tool
            else:
                tool_result = f"Error: Unknown tool {tool_name}"
            # -------------------------------------------

            run_log["steps"].append({
                "type": "tool_call",
                "tool_name": tool_name, 
                "input": tool_input,
                "output": tool_result,
                "tokens": tool_token_cost
            })
            run_log["total_tokens"] += tool_token_cost
            
            # 2. Final Answer Step (Synthesis)
            final_prompt = f"{react_prompt}\n{llm_response_1}\nObservation: {tool_result}\nThought:"
            llm_response_2, tokens_2 = call_llm(self.model_name, final_prompt)
            
            final_answer = self._extract_final_answer(llm_response_2)
            
            run_log["steps"].append({
                "type": "llm_call",
                "model": self.model_name,
                "input": final_prompt,
                "output": final_answer,
                "tokens": tokens_2
            })
            run_log["total_tokens"] += tokens_2
            run_log["final_answer"] = final_answer
        
        else:
            # No tool called (or failed to parse)
            run_log["final_answer"] = self._extract_final_answer(llm_response_1)

        run_log["end_time"] = time.time()
        run_log["latency_seconds"] = run_log["end_time"] - run_log["start_time"]
        return run_log

def main():
    print("="*60)
    print("TESTING LIVE AGENTS (OpenAI & Gemini)")
    print("="*60)
    
    if not openai_client:
        print("Warning: OpenAI API key not configured.")
    if not google_api_key:
        print("Warning: Google API key not configured.")

    # 1. Test OpenAI (Single Tool)
    print("\n--- Testing SingleToolAgent with GPT-4o-mini ---")
    try:
        agent_gpt = SingleToolAgent("gpt-4o-mini")
        log_gpt = agent_gpt.run("What are the risk factors for heart disease?")
        print(f"Final Answer: {log_gpt['final_answer'][:150]}...")
        print(f"Total Tokens: {log_gpt['total_tokens']}")
    except Exception as e:
        print(f"GPT Test Failed: {e}")

    # 2. Test Gemini (Single Tool)
    print("\n--- Testing SingleToolAgent with Gemini-1.5-flash-001 ---")
    try:
        # Using specific version string to avoid 404 errors
        agent_gemini = SingleToolAgent("gemini-1.5-flash-001")
        log_gemini = agent_gemini.run("What are the risk factors for heart disease?")
        print(f"Final Answer: {log_gemini['final_answer'][:150]}...")
        print(f"Total Tokens: {log_gemini['total_tokens']}")
    except Exception as e:
        print(f"Gemini Test Failed: {e}")

if __name__ == "__main__":
    main()