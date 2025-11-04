import json
import re
import time
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

# ---
# IMPORTANT: This script is a "scaffold."
# Real LLM calls are "stubbed" (simulated) to allow for testing
# the agent logic and our custom logger without incurring API costs.
#
# To run the full benchmark, replace all comments marked with
# "REPLACE WITH ACTUAL LLM CALL"
# ---

# Load credentials
load_dotenv() 

# STATIC KNOWLEDGE BASE (For Reproducible Web Search)

STATIC_DOCUMENTS = {
    "diabetes": """
    Diabetes is a chronic health condition affecting how your body turns food into energy.
    Most food is broken down into sugar (glucose) and released into your bloodstream.
    When blood sugar goes up, it signals your pancreas to release insulin.
    Type 1 diabetes is caused by an autoimmune reaction that stops your body from making insulin.
    Type 2 diabetes occurs when your body doesn't use insulin well and can't keep blood sugar at normal levels.
    Prevention includes maintaining a healthy weight, being physically active, and eating healthy foods.
    """,
    "heart_disease": """
    Heart disease refers to several types of heart conditions, with coronary artery disease being most common.
    It can lead to heart attack and is the leading cause of death in the United States.
    Risk factors include high blood pressure, high cholesterol, smoking, diabetes, and obesity.
    Symptoms may include chest pain, shortness of breath, and pain in the neck, jaw, or back.
    Prevention strategies include eating a healthy diet, maintaining healthy weight, exercising regularly,
    managing stress, and avoiding tobacco use.
    """,
    "nutrition": """
    Good nutrition is essential for maintaining health and preventing chronic diseases.
    A balanced diet includes fruits, vegetables, whole grains, lean proteins, and healthy fats.
    Limiting processed foods, added sugars, and excessive sodium is important for health.
    Staying hydrated by drinking adequate water throughout the day is crucial.
    Portion control and mindful eating help maintain a healthy weight.
    Nutritional needs vary by age, gender, activity level, and health conditions.
    """,
    "exercise": """
    Regular physical activity is one of the most important things for health.
    Adults should aim for at least 150 minutes of moderate aerobic activity per week.
    Exercise helps control weight, reduces risk of heart disease, and strengthens bones and muscles.
    It can improve mental health, mood, and ability to do daily activities.
    Types of exercise include aerobic activities, strength training, flexibility exercises, and balance activities.
    Starting slowly and gradually increasing intensity is important for safety.
    """
}

# TASK 2: IMPLEMENT CUSTOM TOOLS (The "Hands")

# A simple set of English stop words to ignore in search
STOP_WORDS = set(["a", "is", "in", "what", "the", "for", "and", "of", "to", "was", "it", "with", "as"])

def web_search(query: str) -> str:
    """
    Static web search tool that searches over local documents.
    This ensures reproducibility - no live API calls.
    """
    query_lower = query.lower()
    
    # Get meaningful, non-stop-words from the query
    query_words = [word.strip("?.,") for word in query_lower.split() if word not in STOP_WORDS]
    
    results = []
    for doc_key, doc_content in STATIC_DOCUMENTS.items():
        doc_key_as_search_term = doc_key.replace("_", " ") # "heart_disease" -> "heart disease"
        
        # 1. Check if the doc_key is a clear match
        if doc_key_as_search_term in query_lower:
            results.append(f"[Document: {doc_key}]\n{doc_content.strip()}")
            continue # Found the best match, go to next document
            
        # 2. If no key match, check if any meaningful query word is in the doc_key
        key_word_match = False
        for word in query_words:
            if word in doc_key: # e.g., 'heart' in 'heart_disease'
                key_word_match = True
                break
        
        if key_word_match:
            results.append(f"[Document: {doc_key}]\n{doc_content.strip()}")
            continue # Found a good match, go to next document

        # 3. If still no match, check if any meaningful query word is in the content
        content_match = False
        for word in query_words:
            if word and word in doc_content.lower(): # 'heart' in '...heart disease...'
                content_match = True
                break
        
        if content_match:
            results.append(f"[Document: {doc_key}]\n{doc_content.strip()}")
            
    if results:
        # Use set to remove duplicate doc matches
        return "\n\n".join(list(dict.fromkeys(results)))
    else:
        return "No relevant documents found for your query."

def content_extractor(text: str) -> str:
    """
    Extracts and summarizes key information from text using an LLM.
    (Stubbed for now)
    """
    prompt = f"You are a content extraction specialist...{text}..."
    
    # --- REPLACE WITH ACTUAL LLM CALL ---
    # response = GeminiFlash.generate_content(prompt)
    # summary = response.text
    # token_count = response.token_count 
    
    
    # For demonstration purposes:
    summary = f"[EXTRACTED CONTENT]\nKey points identified:\n"
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    for i, sentence in enumerate(sentences[:3], 1):
        summary += f"{i}. {sentence}.\n"
        
    # We return both the output and a simulated token count
    return summary, 150 # (output, token_count)


def quiz_generator(text: str) -> str:
    """
    Generates a multiple-choice quiz from provided text using an LLM.
    (Stubbed for now)
    """
    prompt = f"You are a quiz generator...{text}..."
    
    # --- REPLACE WITH ACTUAL LLM CALL ---
    # response = GeminiFlash.generate_content(prompt)
    # quiz = response.text
    # token_count = response.token_count 
    
    
    # For demonstration purposes:
    quiz = """Question 1: What is a key prevention strategy mentioned?
A) Taking daily medication
B) Maintaining a healthy weight
...
Correct Answer: B"""
    
    # We return both the output and a simulated token count
    return quiz, 200 # (output, token_count)

class NoToolAgent:
    """
    Simplest agent - directly queries LLM without any tools.
    Returns a dictionary log of the run.
    """
    
    def __init__(self, model_name: str = "no-tool-agent-stubbed"):
        self.model_name = model_name
    
    def run(self, prompt: str) -> Dict:
        """
        Execute the agent and return a log of the run.
        """
        run_log = {
            "agent_type": "no_tool",
            "start_time": time.time(),
            "steps": [],
            "total_tokens": 0,
            "final_answer": None,
            "latency_seconds": 0
        }
        
        # --- REPLACE WITH ACTUAL LLM CALL ---
        # response = GeminiFlash.generate_content(prompt)
        # final_answer = response.text
        # token_count = response.token_count
        
        
        # Simulated response
        final_answer = f"Based on my internal knowledge, here's my response to: {prompt}\n[...]"
        token_count = 50 # Simulated token count
        
        run_log["steps"].append({
            "type": "llm_call",
            "model": self.model_name,
            "input": prompt,
            "output": final_answer,
            "tokens": token_count
        })
        run_log["total_tokens"] = token_count
        run_log["final_answer"] = final_answer
        run_log["end_time"] = time.time()
        run_log["latency_seconds"] = run_log["end_time"] - run_log["start_time"]
        
        return run_log


class SingleToolAgent:
    """
    ReAct-style agent with web_search tool.
    Returns a dictionary log of the run.
    """
    
    def __init__(self, model_name: str = "single-tool-agent-stubbed"):
        self.model_name = model_name
        self.tool_definition = "..." 
    
    def _parse_tool_call(self, llm_response: str) -> Optional[Dict]:
        action_match = re.search(r'Action:\s*(\w+)', llm_response)
        input_match = re.search(r'Action Input:\s*["\']?([^"\']+)["\']?', llm_response)
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

        react_prompt = f"You are a helpful assistant...Question: {prompt}\nThought:"

        # --- REPLACE WITH ACTUAL LLM CALL (Step 1) ---
        # llm_response_1 = GeminiFlash.generate_content(react_prompt).text
        # token_count_1 = ...
        
        
        # Simulated reasoning response
        llm_response_1 = f"I should search for information...\nAction: web_search\nAction Input: {prompt}"
        token_count_1 = 70 # Simulated
        
        run_log["steps"].append({
            "type": "llm_call",
            "model": self.model_name,
            "input": react_prompt,
            "output": llm_response_1,
            "tokens": token_count_1
        })
        run_log["total_tokens"] += token_count_1
        
        tool_call = self._parse_tool_call(llm_response_1)
        
        if tool_call and tool_call["tool"] == "web_search":
            search_result = web_search(tool_call["query"])
            
            run_log["steps"].append({
                "type": "tool_call",
                "tool_name": "web_search",
                "input": tool_call["query"],
                "output": search_result
            })
            
            final_prompt = f"{react_prompt}\n{llm_response_1}\nObservation: {search_result}\nThought:"
            
            # --- REPLACE WITH ACTUAL LLM CALL (Step 2) ---
            # llm_response_2 = GeminiFlash.generate_content(final_prompt).text
            # token_count_2 = ...
            # final_answer = ... (parse llm_response_2 for "Final Answer:")
            
            
            # Simulated final response
            final_answer = f"Based on the search results, {prompt.lower()}\n\nKey information: {search_result[:200]}..."
            token_count_2 = 30 # Simulated
            
            run_log["steps"].append({
                "type": "llm_call",
                "model": self.model_name,
                "input": final_prompt,
                "output": final_answer,
                "tokens": token_count_2
            })
            run_log["total_tokens"] += token_count_2
            run_log["final_answer"] = final_answer
            
        else:
            run_log["final_answer"] = llm_response_1 
        
        run_log["end_time"] = time.time()
        run_log["latency_seconds"] = run_log["end_time"] - run_log["start_time"]
        return run_log


class MultiToolAgent:
    """
    ReAct-style agent with multiple tools.
    Returns a dictionary log of the run.
    """
    
    def __init__(self, model_name: str = "multi-tool-agent-stubbed"):
        self.model_name = model_name
        self.tool_definitions = "..." 
    
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
        
        tools_desc = "..." 
        react_prompt = f"You are a helpful assistant...{tools_desc}\nQuestion: {prompt}\nThought:"

        # REPLACE WITH ACTUAL LLM CALL 
        # llm_response_1 = GeminiFlash.generate_content(react_prompt).text
        # token_count_1 = ...
        
        
        # Simulated reasoning
        token_count_1 = 80 # Simulated
        if "quiz" in prompt.lower():
            llm_response_1 = f"Action: quiz_generator\nAction Input: Sample text..."
        else:
            llm_response_1 = f"Action: content_extractor\nAction Input: Sample text..."
            
        run_log["steps"].append({
            "type": "llm_call",
            "model": self.model_name,
            "input": react_prompt,
            "output": llm_response_1,
            "tokens": token_count_1
        })
        run_log["total_tokens"] += token_count_1

        tool_call = self._parse_tool_call(llm_response_1)
        
        if tool_call:
            tool_name = tool_call["tool"]
            tool_input = tool_call["input"]
            tool_result = ""
            tool_token_cost = 0

            if tool_name == "content_extractor":
                tool_result, tool_token_cost = content_extractor(tool_input)
            elif tool_name == "quiz_generator":
                tool_result, tool_token_cost = quiz_generator(tool_input)
            
            run_log["steps"].append({
                "type": "tool_call",
                "tool_name": tool_name,
                "input": tool_input,
                "output": tool_result
            })
            run_log["total_tokens"] += tool_token_cost # Add tokens if tool was an LLM call

            final_prompt = f"{react_prompt}\n{llm_response_1}\nObservation: {tool_result}\nThought:"
            
            # --- REPLACE WITH ACTUAL LLM CALL (Step 2) ---
            # llm_response_2 = GeminiFlash.generate_content(final_prompt).text
            # token_count_2 = ...
            # final_answer = ... (parse final_response)
            
            
            final_answer = f"Here's the result:\n\n{tool_result}"
            token_count_2 = 25 # Simulated
            
            run_log["steps"].append({
                "type": "llm_call",
                "model": self.model_name,
                "input": final_prompt,
                "output": final_answer,
                "tokens": token_count_2
            })
            run_log["total_tokens"] += token_count_2
            run_log["final_answer"] = final_answer
        
        else:
            run_log["final_answer"] = llm_response_1

        run_log["end_time"] = time.time()
        run_log["latency_seconds"] = run_log["end_time"] - run_log["start_time"]
        return run_log

# USAGE EXAMPLES (Demonstration of the scaffold)

def main():
    """Demonstrate all three agents."""
    
    print("=" * 80)
    print("AGENTIC SYSTEM DEMONSTRATION")
    print("=" * 80)
    
    test_prompts = [
        "What are the risk factors for heart disease?",
        "Extract key information about diabetes prevention",
        "Generate a quiz about nutrition and healthy eating"
    ]
    
    no_tool = NoToolAgent()
    single_tool = SingleToolAgent()
    multi_tool = MultiToolAgent()
    
    # Test No-Tool Agent
    print("\n[NO-TOOL AGENT RUN]")
    log1 = no_tool.run(test_prompts[0])
    print(json.dumps(log1, indent=2))

    # Test Single-Tool Agent
    print("\n[SINGLE-TOOL AGENT RUN]")
    log2 = single_tool.run(test_prompts[0])
    print(json.dumps(log2, indent=2))

    # Test Multi-Tool Agent
    print("\n[MULTI-TOOL AGENT RUN (QUIZ)]")
    log3 = multi_tool.run(test_prompts[2])
    print(json.dumps(log3, indent=2))
    
    print("\n" + "=" * 80)
    print("All runs completed and logs generated successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()