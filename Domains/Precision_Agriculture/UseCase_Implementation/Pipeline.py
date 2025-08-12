######################################################################
# Free LLM Integration for Blueberry Harvest Analysis
# Supports multiple free LLM options without requiring paid API keys
######################################################################

import json
import requests
from datetime import datetime

######################################################################
# Local Model Integration using Transformers Library
######################################################################

class LocalLLMProcessor:
    """
    Handles local language model processing using free transformer models.
    Falls back to rule-based analysis if models are unavailable.
    """
    
    def __init__(self):
        self.is_model_ready = False
        self.tokenizer = None
        self.language_model = None
        self.initialize_local_model()
    
    def initialize_local_model(self):
        """
        Attempts to load a free language model that doesn't require authentication.
        Uses Google Flan-T5 as the primary choice due to its instruction-following capabilities.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_identifier = "google/flan-t5-base"
            
            print(f"Loading {model_identifier} model (no authentication required)...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_identifier)
            self.language_model = AutoModelForCausalLM.from_pretrained(model_identifier)
            
            self.is_model_ready = True
            print("Local model loaded successfully")
            
        except Exception as error:
            print(f"Could not load transformer model: {error}")
            print("Switching to rule-based analysis instead")
            self.is_model_ready = False
    
    def create_analysis_response(self, input_prompt, token_limit=200):
        """
        Generates harvest analysis response using either the loaded model or rule-based logic.
        """
        if not self.is_model_ready:
            return self.perform_rule_based_analysis(input_prompt)
        
        try:
            simplified_input = self.prepare_model_prompt(input_prompt)
            
            model_inputs = self.tokenizer(simplified_input, return_tensors="pt", 
                                        max_length=512, truncation=True)
            
            with torch.no_grad():
                model_outputs = self.language_model.generate(
                    **model_inputs,
                    max_length=token_limit,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
            return generated_text
            
        except Exception as error:
            print(f"Model generation failed: {error}")
            return self.perform_rule_based_analysis(input_prompt)
    
    def prepare_model_prompt(self, original_input):
        """
        Simplifies the input prompt for better compatibility with T5 model architecture.
        Extracts key ripeness data and creates focused instructions.
        """
        prompt_lines = original_input.split('\n')
        
        ready_percentage = 26.5  # Default value
        overripe_percentage = 3.5  # Default value
        
        for line in prompt_lines:
            if 'Ready to harvest (R4):' in line:
                try:
                    ready_percentage = float(line.split(':')[1].strip().rstrip('%'))
                except ValueError:
                    continue
            elif 'Overripe berries (R5):' in line:
                try:
                    overripe_percentage = float(line.split(':')[1].strip().rstrip('%'))
                except ValueError:
                    continue
        
        optimized_prompt = f"Analyze blueberry harvest with {ready_percentage}% ready berries and {overripe_percentage}% overripe. Recommend workforce and harvesting strategy."
        return optimized_prompt
    
    def perform_rule_based_analysis(self, input_prompt):
        """
        Provides intelligent harvest recommendations using algorithmic analysis
        when language models are not available.
        """
        print("Using algorithmic analysis for harvest recommendations")
        
        ######################################################################
        # Extract ripeness data from the input prompt
        ######################################################################
        prompt_lines = input_prompt.split('\n')
        ready_berries_percent = 26.5
        overripe_berries_percent = 3.5
        
        for line in prompt_lines:
            if 'Ready to harvest (R4):' in line:
                try:
                    ready_berries_percent = float(line.split(':')[1].strip().rstrip('%'))
                except ValueError:
                    continue
            elif 'Overripe berries (R5):' in line:
                try:
                    overripe_berries_percent = float(line.split(':')[1].strip().rstrip('%'))
                except ValueError:
                    continue
        
        ######################################################################
        # Apply harvest decision logic based on ripeness levels
        ######################################################################
        if overripe_berries_percent > 10:
            harvest_urgency = "high"
            recommended_workers = max(6, int(ready_berries_percent / 3))
            harvest_strategy = "immediate harvest to minimize waste"
        elif ready_berries_percent > 30:
            harvest_urgency = "medium"
            recommended_workers = max(4, int(ready_berries_percent / 4))
            harvest_strategy = "begin systematic harvest"
        else:
            harvest_urgency = "low"
            recommended_workers = max(2, int(ready_berries_percent / 5))
            harvest_strategy = "monitor and prepare for harvest"
        
        ######################################################################
        # Calculate expected yield and quality metrics
        ######################################################################
        estimated_yield = int(ready_berries_percent * 25)
        
        analysis_report = f"""
HARVEST ANALYSIS REPORT

WORKFORCE PLANNING: Deploy {recommended_workers} workers tomorrow
HARVEST URGENCY: {harvest_urgency.upper()} priority
YIELD ESTIMATION: Expected {estimated_yield} lbs harvest
QUALITY STRATEGY: {"Speed-focused harvesting" if harvest_urgency == "high" else "Selective picking approach"}
WASTE MANAGEMENT: {overripe_berries_percent}% waste expected
TIMING RECOMMENDATIONS: {"Immediate action required" if harvest_urgency == "high" else "2-3 day optimal window"}

RECOMMENDATIONS:
- {harvest_strategy}
- Monitor overripe levels closely (currently {overripe_berries_percent}%)
- Focus on high-density ready berry areas first
- {"Weather permitting, extend harvest hours" if harvest_urgency == "high" else "Standard harvest schedule recommended"}
- Quality control: {"Prioritize speed over perfection" if harvest_urgency == "high" else "Maintain selective picking standards"}
"""
        
        return analysis_report.strip()

######################################################################
# Ollama Integration for Local Model Serving
######################################################################

class OllamaModelManager:
    """
    Manages Ollama-served models for local LLM processing.
    Handles model selection and installation automatically.
    """
    
    def __init__(self):
        self.ollama_endpoint = "http://localhost:11434"
        self.supported_models = [
            "phi",           # Microsoft Phi - compact and efficient
            "mistral",       # Mistral 7B - open source general model
            "codellama",     # Code Llama - good for technical analysis
            "orca-mini",     # Orca Mini - lightweight alternative
            "vicuna"         # Vicuna - community-trained model
        ]
        self.active_model = None
        self.find_available_model()
    
    def find_available_model(self):
        """
        Checks which supported models are already installed and ready to use.
        """
        try:
            model_list_response = requests.get(f"{self.ollama_endpoint}/api/tags")
            if model_list_response.status_code == 200:
                installed_models = model_list_response.json()
                installed_model_names = [model['name'].split(':')[0] 
                                       for model in installed_models.get('models', [])]
                
                for model in self.supported_models:
                    if model in installed_model_names:
                        self.active_model = model
                        print(f"Using {model} model for analysis")
                        return
                
                print("No suitable models found. Installing Phi model (lightweight option)...")
                self.setup_model("phi")
            else:
                print("Ollama service is not running")
                
        except requests.exceptions.ConnectionError:
            print("Ollama is not available on this system")
    
    def setup_model(self, model_name):
        """
        Downloads and installs a specified model through Ollama.
        """
        try:
            print(f"Installing {model_name} model...")
            install_response = requests.post(f"{self.ollama_endpoint}/api/pull", 
                                           json={"name": model_name})
            if install_response.status_code == 200:
                self.active_model = model_name
                print(f"{model_name} model installed successfully")
            else:
                print(f"Failed to install {model_name} model")
        except Exception as error:
            print(f"Model installation error: {error}")
    
    def generate_analysis(self, prompt_text):
        """
        Generates harvest analysis using the active Ollama model.
        """
        if not self.active_model:
            return None
        
        try:
            request_payload = {
                "model": self.active_model,
                "prompt": prompt_text,
                "stream": False
            }
            
            generation_response = requests.post(f"{self.ollama_endpoint}/api/generate", 
                                              json=request_payload)
            
            if generation_response.status_code == 200:
                return generation_response.json().get("response", "")
            else:
                return None
                
        except Exception as error:
            print(f"Response generation error: {error}")
            return None

######################################################################
# Groq API Integration with Free Tier Support
######################################################################

class GroqAPIClient:
    """
    Handles Groq API integration with support for free tier usage.
    Provides access to fast inference with daily token limits.
    """
    
    def __init__(self, api_key=None):
        self.groq_api_key = api_key
        self.api_endpoint = "https://api.groq.com/openai/v1"
        
        if not api_key:
            print("No Groq API key provided")
            print("Free API key available at: https://console.groq.com/")
            print("Free tier includes 14,400 tokens per day")
    
    def generate_analysis(self, prompt_text):
        """
        Generates harvest analysis using Groq's API with free tier models.
        """
        if not self.groq_api_key:
            return None
        
        try:
            request_headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            request_data = {
                "model": "llama2-70b-4096",
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            api_response = requests.post(f"{self.api_endpoint}/chat/completions", 
                                       headers=request_headers, json=request_data)
            
            if api_response.status_code == 200:
                response_data = api_response.json()
                return response_data["choices"][0]["message"]["content"]
            else:
                print(f"Groq API returned error: {api_response.status_code}")
                return None
                
        except Exception as error:
            print(f"Groq API error: {error}")
            return None

######################################################################
# Main Pipeline Integration
######################################################################

def execute_harvest_analysis_pipeline():
    """
    Runs the complete harvest analysis pipeline using available LLM options.
    Tries multiple LLM services in order of preference and falls back as needed.
    """
    
    print("BLUEBERRY HARVEST PLANNING WITH FREE LLM INTEGRATION")
    print("=" * 60)
    
    ######################################################################
    # Initialize available LLM services in order of preference
    ######################################################################
    llm_services = [
        ("Local Transformer Models", LocalLLMProcessor()),
        ("Ollama Local Models", OllamaModelManager()),
        ("Groq API Service", GroqAPIClient())
    ]
    
    selected_service = None
    
    print("\nChecking available LLM services...")
    for service_name, service_instance in llm_services:
        if isinstance(service_instance, LocalLLMProcessor):
            selected_service = service_instance
            print(f"Using {service_name}")
            break
        elif isinstance(service_instance, OllamaModelManager) and service_instance.active_model:
            selected_service = service_instance
            print(f"Using {service_name} - {service_instance.active_model}")
            break
    
    if not selected_service:
        print("Using rule-based analysis as fallback option...")
        selected_service = LocalLLMProcessor()
    
    ######################################################################
    # Load computer vision analysis data
    ######################################################################
    print("\nLoading computer vision analysis data...")
    try:
        with open('cv_dummy_output.json', 'r') as data_file:
            cv_analysis_data = json.load(data_file)
        print(f"Loaded analysis for {cv_analysis_data['total_berries_detected']:,} berries")
    except FileNotFoundError:
        print("Error: cv_dummy_output.json file not found")
        return
    
    ######################################################################
    # Generate ripeness forecast for tomorrow
    ######################################################################
    print("\nGenerating ripeness forecast...")
    current_distribution = cv_analysis_data['ripeness_distribution']
    
    # Apply simple growth model for next day prediction
    forecasted_distribution = [percentage + (percentage * 0.1) for percentage in current_distribution]
    distribution_total = sum(forecasted_distribution)
    forecasted_distribution = [round(forecast/distribution_total * 100, 1) 
                             for forecast in forecasted_distribution]
    
    ######################################################################
    # Create comprehensive analysis prompt
    ######################################################################
    analysis_prompt = f"""You are a professional blueberry harvest consultant. Analyze this field data:

CURRENT RIPENESS DISTRIBUTION:
- Ready to harvest (R4): {current_distribution[3]}%
- Overripe berries (R5): {current_distribution[4]}%

TOMORROW'S FORECAST:
- Ready to harvest (R4): {forecasted_distribution[3]}%
- Overripe berries (R5): {forecasted_distribution[4]}%

FARM SPECIFICATIONS: 15,000 blueberry plants, 8 workers available, 200 lbs per worker per day capacity

Provide specific recommendations for harvest planning including workforce allocation and timing strategy."""
    
    ######################################################################
    # Generate LLM analysis response
    ######################################################################
    print("\nGenerating LLM analysis...")
    
    if isinstance(selected_service, LocalLLMProcessor):
        analysis_response = selected_service.create_analysis_response(analysis_prompt)
    elif isinstance(selected_service, OllamaModelManager):
        analysis_response = selected_service.generate_analysis(analysis_prompt)
    elif isinstance(selected_service, GroqAPIClient):
        analysis_response = selected_service.generate_analysis(analysis_prompt)
    else:
        analysis_response = None
    
    if analysis_response:
        print("\n" + "=" * 60)
        print("LLM HARVEST ANALYSIS RESULTS")
        print("=" * 60)
        
        print(f"Analysis Date: {cv_analysis_data['analysis_date']}")
        print(f"Total Berries: {cv_analysis_data['total_berries_detected']:,}")
        print(f"Ready for Harvest: {current_distribution[3]}%")
        print(f"Overripe Berries: {current_distribution[4]}%")
        
        print(f"\nLLM RECOMMENDATIONS:")
        print("-" * 40)
        print(analysis_response)
        print("-" * 40)
        
        ######################################################################
        # Save complete analysis results
        ######################################################################
        complete_results = {
            "analysis_date": cv_analysis_data['analysis_date'],
            "cv_data": cv_analysis_data,
            "llm_response": analysis_response,
            "model_used": type(selected_service).__name__
        }
        
        with open('free_llm_analysis.json', 'w') as results_file:
            json.dump(complete_results, results_file, indent=2)
        
        print(f"\nAnalysis results saved to: free_llm_analysis.json")
        
    else:
        print("No LLM response could be generated")

######################################################################
# Setup and Configuration Information
######################################################################

def display_free_llm_options():
    """
    Displays available free LLM options and setup instructions for users.
    """
    
    print("FREE LLM OPTIONS FOR BLUEBERRY ANALYSIS")
    print("=" * 40)
    
    print("\nOPTION 1: RULE-BASED ANALYSIS (IMMEDIATE USE)")
    print("   - Works without any setup")
    print("   - Intelligent algorithmic recommendations")
    print("   - No internet connection required")
    
    print("\nOPTION 2: OLLAMA LOCAL MODELS")
    print("   - Download from: https://ollama.ai/download")
    print("   - Start service: ollama serve")
    print("   - Install model: ollama pull phi")
    
    print("\nOPTION 3: GROQ API (FREE TIER)")
    print("   - Register at: https://console.groq.com/")
    print("   - Obtain free API key (14,400 tokens daily)")
    print("   - Add API key to configuration")
    
    print("\nRECOMMENDATION: Start with Option 1 for immediate results")

######################################################################
# Main Execution
######################################################################

if __name__ == "__main__":
    display_free_llm_options()
    
    print("\n" + "="*60)
    user_choice = input("Would you like to run the harvest analysis pipeline? (y/n): ")
    
    if user_choice.lower() == 'y':
        execute_harvest_analysis_pipeline()
    else:
        print("Pipeline ready to run when needed")
