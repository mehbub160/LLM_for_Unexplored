# ============================================================================
# FREE LLM ALTERNATIVES - NO AUTHENTICATION REQUIRED
# Using Microsoft DialoGPT, Google Flan-T5, or Ollama alternatives
# ============================================================================

import json
import requests
from datetime import datetime

# ============================================================================
# 1. MICROSOFT DIALOGPT (FREE, NO AUTH)
# ============================================================================

class FreeLLMIntegration:
    """Free LLM integration using publicly available models"""
    
    def __init__(self):
        self.model_loaded = False
        self.tokenizer = None
        self.model = None
        self.load_free_model()
    
    def load_free_model(self):
        """Load a free model that doesn't require authentication"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Try Google Flan-T5 first (text-to-text, good for instructions)
            model_name = "google/flan-t5-base"
            
            print(f"📥 Loading {model_name} (free, no auth required)...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            self.model_loaded = True
            print("✅ Free model loaded successfully!")
            
        except Exception as e:
            print(f"⚠️  Transformers error: {e}")
            print("   Falling back to rule-based analysis...")
            self.model_loaded = False
    
    def generate_response(self, prompt, max_tokens=200):
        """Generate response using free model"""
        if not self.model_loaded:
            return self.rule_based_analysis(prompt)
        
        try:
            # Simplify prompt for T5 model
            simplified_prompt = self.create_simple_prompt(prompt)
            
            inputs = self.tokenizer(simplified_prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            print(f"⚠️  Model generation error: {e}")
            return self.rule_based_analysis(prompt)
    
    def create_simple_prompt(self, original_prompt):
        """Create simplified prompt for T5 model"""
        # Extract key information from original prompt
        lines = original_prompt.split('\n')
        
        r4_current = 26.5  # Default
        r5_current = 3.5   # Default
        
        for line in lines:
            if 'Ready to harvest (R4):' in line:
                try:
                    r4_current = float(line.split(':')[1].strip().rstrip('%'))
                except:
                    pass
            elif 'Overripe berries (R5):' in line:
                try:
                    r5_current = float(line.split(':')[1].strip().rstrip('%'))
                except:
                    pass
        
        # Create simple instruction for T5
        simple_prompt = f"Analyze blueberry harvest: {r4_current}% ready, {r5_current}% overripe. Recommend workers and strategy."
        return simple_prompt
    
    def rule_based_analysis(self, prompt):
        """Fallback rule-based analysis if models fail"""
        print("🤖 Using rule-based analysis (no LLM required)")
        
        # Extract ripeness data from prompt
        lines = prompt.split('\n')
        r4_current = 26.5  # Default
        r5_current = 3.5   # Default
        
        for line in lines:
            if 'Ready to harvest (R4):' in line:
                try:
                    r4_current = float(line.split(':')[1].strip().rstrip('%'))
                except:
                    pass
            elif 'Overripe berries (R5):' in line:
                try:
                    r5_current = float(line.split(':')[1].strip().rstrip('%'))
                except:
                    pass
        
        # Rule-based recommendations
        if r5_current > 10:
            urgency = "high"
            workers = max(6, int(r4_current / 3))
            strategy = "immediate harvest to minimize waste"
        elif r4_current > 30:
            urgency = "medium"
            workers = max(4, int(r4_current / 4))
            strategy = "begin systematic harvest"
        else:
            urgency = "low"
            workers = max(2, int(r4_current / 5))
            strategy = "monitor and prepare for harvest"
        
        # Calculate yield estimate
        yield_estimate = int(r4_current * 25)  # Rough estimate
        
        # Generate structured response
        response = f"""
HARVEST ANALYSIS REPORT:

1. WORKFORCE PLANNING: Deploy {workers} workers tomorrow
2. HARVEST URGENCY: {urgency.upper()} priority
3. YIELD ESTIMATION: Expected {yield_estimate} lbs harvest
4. QUALITY STRATEGY: {"Speed-focused" if urgency == "high" else "Selective picking"}
5. WASTE MANAGEMENT: {r5_current}% waste expected
6. TIMING RECOMMENDATIONS: {"Immediate action" if urgency == "high" else "2-3 day optimal window"}

RECOMMENDATIONS:
- {strategy}
- Monitor R5 levels closely (currently {r5_current}%)
- Focus on high-density R4 areas first
- {"Weather permitting, extend harvest hours" if urgency == "high" else "Standard harvest schedule recommended"}
- Quality control: {"Prioritize speed over perfection" if urgency == "high" else "Maintain selective picking standards"}
"""
        
        return response.strip()

# ============================================================================
# 2. OLLAMA WITH ALTERNATIVE MODELS (FREE)
# ============================================================================

class OllamaFreeModels:
    """Ollama with free alternative models"""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.available_models = [
            "phi",           # Microsoft's Phi model - small and fast
            "mistral",       # Mistral 7B - open source
            "codellama",     # Code Llama - free variant
            "orca-mini",     # Orca Mini - lightweight
            "vicuna"         # Vicuna - open source
        ]
        self.selected_model = None
        self.check_available_models()
    
    def check_available_models(self):
        """Check which models are available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                installed = response.json()
                installed_names = [model['name'].split(':')[0] for model in installed.get('models', [])]
                
                # Find first available model
                for model in self.available_models:
                    if model in installed_names:
                        self.selected_model = model
                        print(f"✅ Using {model} model")
                        return
                
                print("⚠️  No suitable models found. Installing phi (lightweight)...")
                self.install_model("phi")
            else:
                print("❌ Ollama not running")
                
        except requests.exceptions.ConnectionError:
            print("❌ Ollama not available")
    
    def install_model(self, model_name):
        """Install a free model"""
        try:
            print(f"📥 Installing {model_name}...")
            response = requests.post(f"{self.base_url}/api/pull", 
                                   json={"name": model_name})
            if response.status_code == 200:
                self.selected_model = model_name
                print(f"✅ {model_name} installed")
            else:
                print(f"❌ Failed to install {model_name}")
        except Exception as e:
            print(f"❌ Installation error: {e}")
    
    def generate_response(self, prompt):
        """Generate response using free Ollama model"""
        if not self.selected_model:
            return None
        
        try:
            payload = {
                "model": self.selected_model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return None
                
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return None

# ============================================================================
# 3. GROQ API (FREE TIER)
# ============================================================================

class GroqAPI:
    """Groq API with free tier"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
        
        if not api_key:
            print("⚠️  No Groq API key provided")
            print("   Get free API key at: https://console.groq.com/")
            print("   Free tier: 14,400 tokens/day")
    
    def generate_response(self, prompt):
        """Generate response using Groq API"""
        if not self.api_key:
            return None
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama2-70b-4096",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(f"{self.base_url}/chat/completions", 
                                   headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"❌ Groq API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Groq error: {e}")
            return None

# ============================================================================
# 4. INTEGRATED PIPELINE WITH FREE ALTERNATIVES
# ============================================================================

def run_pipeline_with_free_llm():
    """Run pipeline with free LLM alternatives"""
    
    print("🫐 BLUEBERRY HARVEST PLANNING WITH FREE LLM")
    print("=" * 60)
    
    # Try different LLM options in order of preference
    llm_options = [
        ("Rule-based Analysis", FreeLLMIntegration()),
        ("Ollama Free Models", OllamaFreeModels()),
        ("Groq API (if key available)", GroqAPI())
    ]
    
    selected_llm = None
    
    print("\n1. Checking available LLM options...")
    for name, llm in llm_options:
        if isinstance(llm, FreeLLMIntegration):
            selected_llm = llm
            print(f"✅ Using {name}")
            break
        elif isinstance(llm, OllamaFreeModels) and llm.selected_model:
            selected_llm = llm
            print(f"✅ Using {name} - {llm.selected_model}")
            break
    
    if not selected_llm:
        print("Using rule-based analysis as fallback...")
        selected_llm = FreeLLMIntegration()
    
    # Load CV data
    print("\n2. Loading CV analysis data...")
    try:
        with open('cv_dummy_output.json', 'r') as f:
            cv_data = json.load(f)
        print(f"✅ Loaded: {cv_data['total_berries_detected']:,} berries")
    except FileNotFoundError:
        print("❌ cv_dummy_output.json not found!")
        return
    
    # Generate forecast
    print("\n3. Generating forecast...")
    current_ripeness = cv_data['ripeness_distribution']
    forecast = [r + (r * 0.1) for r in current_ripeness]
    total = sum(forecast)
    forecast = [round(f/total * 100, 1) for f in forecast]
    
    # Create prompt
    prompt = f"""You are a blueberry harvest consultant. Analyze this data:

CURRENT RIPENESS:
- Ready to harvest (R4): {current_ripeness[3]}%
- Overripe berries (R5): {current_ripeness[4]}%

TOMORROW'S FORECAST:
- Ready to harvest (R4): {forecast[3]}%
- Overripe berries (R5): {forecast[4]}%

FARM: 15,000 plants, 8 workers available, 200 lbs/worker/day capacity

Provide specific recommendations for harvest planning."""
    
    # Get LLM response
    print("\n4. Getting LLM analysis...")
    response = selected_llm.generate_response(prompt)
    
    if response:
        print("\n" + "=" * 60)
        print("🤖 LLM HARVEST ANALYSIS")
        print("=" * 60)
        
        print(f"📅 Date: {cv_data['analysis_date']}")
        print(f"🫐 Berries: {cv_data['total_berries_detected']:,}")
        print(f"📊 R4 (Ready): {current_ripeness[3]}%")
        print(f"📊 R5 (Overripe): {current_ripeness[4]}%")
        
        print(f"\n🤖 LLM RECOMMENDATIONS:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Save results
        results = {
            "analysis_date": cv_data['analysis_date'],
            "cv_data": cv_data,
            "llm_response": response,
            "model_used": type(selected_llm).__name__
        }
        
        with open('free_llm_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: free_llm_analysis.json")
        
    else:
        print("❌ No LLM response available")

# ============================================================================
# 5. SIMPLE SETUP INSTRUCTIONS
# ============================================================================

def print_free_options():
    """Print free LLM options"""
    
    print("🆓 FREE LLM OPTIONS")
    print("=" * 40)
    
    print("\n✅ OPTION 1: RULE-BASED (NO SETUP)")
    print("   • Works immediately")
    print("   • Smart algorithms")
    print("   • No internet required")
    
    print("\n✅ OPTION 2: OLLAMA FREE MODELS")
    print("   • Install: https://ollama.ai/download")
    print("   • Run: ollama serve")
    print("   • Install: ollama pull phi")
    
    print("\n✅ OPTION 3: GROQ API (FREE TIER)")
    print("   • Sign up: https://console.groq.com/")
    print("   • Get API key (free 14,400 tokens/day)")
    print("   • Add key to script")
    
    print("\n🚀 RECOMMENDED: Start with Option 1 (works immediately)")

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print_free_options()
    
    print("\n" + "="*60)
    response = input("Run the pipeline now? (y/n): ")
    
    if response.lower() == 'y':
        run_pipeline_with_free_llm()
    else:
        print("Ready when you are!")