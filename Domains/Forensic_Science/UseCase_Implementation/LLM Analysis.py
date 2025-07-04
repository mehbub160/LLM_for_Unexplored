# ============================================================================
# SIMPLE LLM-BASED LIE DETECTION USING REAL LLM MODELS
# Direct LLM analysis without complex expert systems
# ============================================================================

import json
import requests
from datetime import datetime
from typing import Dict, List

# ============================================================================
# 1. SIMPLE LLM LIE DETECTION ENGINE
# ============================================================================

class SimpleLLMLieDetector:
    """Simple LLM-based lie detection using actual language models"""
    
    def __init__(self, llm_type="transformers"):
        self.llm_type = llm_type
        print(f"🧠 Initializing {llm_type.upper()} LLM Lie Detector...")
        
        if llm_type == "transformers":
            self.setup_transformers()
        elif llm_type == "ollama":
            self.setup_ollama()
        else:
            print("❌ Unsupported LLM type. Use 'transformers' or 'ollama'")
    
    def setup_transformers(self):
        """Setup Hugging Face Transformers"""
        try:
            from transformers import pipeline
            print("📥 Loading Transformers model...")
            
            # Use a text generation model for analysis
            self.llm = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium",
                device=-1  # Use CPU
            )
            print("✅ Transformers model loaded successfully")
            
        except ImportError:
            print("❌ Please install transformers: pip install transformers torch")
            exit(1)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit(1)
    
    def setup_ollama(self):
        """Setup Ollama LLM"""
        try:
            print("🦙 Connecting to Ollama...")
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                if models:
                    # Use first available model
                    self.model_name = models[0]['name']
                    print(f"✅ Using Ollama model: {self.model_name}")
                else:
                    print("❌ No Ollama models found. Install one: ollama pull llama2")
                    exit(1)
            else:
                print("❌ Ollama server not responding")
                exit(1)
                
        except requests.exceptions.RequestException:
            print("❌ Ollama server not running. Start with: ollama serve")
            exit(1)
    
    def analyze_conversation(self, case_data):
        """Main analysis function using LLM"""
        
        print("🔍 Analyzing conversation with LLM...")
        
        # Step 1: Create LLM prompt from case data
        prompt = self.create_analysis_prompt(case_data)
        
        # Step 2: Get LLM analysis
        if self.llm_type == "transformers":
            llm_response = self.analyze_with_transformers(prompt)
        elif self.llm_type == "ollama":
            llm_response = self.analyze_with_ollama(prompt)
        
        # Step 3: Structure the LLM output
        analysis_result = self.structure_llm_output(llm_response, case_data)
        
        return analysis_result
    
    def create_analysis_prompt(self, case_data):
        """Create focused prompt for LLM analysis"""
        
        # Extract key information
        case_info = case_data["case_information"]
        conversation = case_data["conversation_transcript"]
        facts = case_data["established_facts"]
        
        # Format conversation
        conversation_text = ""
        for msg in conversation:
            if msg["speaker"] == "Subject":
                conversation_text += f"SUSPECT: {msg['content']}\n"
            else:
                conversation_text += f"DETECTIVE: {msg['content']}\n"
        
        # Create prompt
        prompt = f"""You are a forensic psychologist expert in detecting deception through conversation analysis.

CASE: {case_info['case_type'].title()} - {case_info['incident_description']}

ESTABLISHED FACTS:
- Incident occurred at {case_info['location']} on {case_info['incident_date']} at {case_info['incident_time']}
- Key evidence: {', '.join(facts['physical_evidence'][:3])}

INTERROGATION TRANSCRIPT:
{conversation_text}

ANALYSIS TASK:
Analyze this interrogation for signs of deception. Focus on:

1. DECEPTION INDICATORS: Language patterns suggesting lies or fabrication
2. CONTRADICTIONS: Statements that conflict with evidence or each other
3. BEHAVIORAL PATTERNS: Psychological signs of deception
4. OVERALL ASSESSMENT: High/Medium/Low likelihood of deception

Provide specific examples from the conversation and explain your reasoning.

ANALYSIS:"""

        return prompt
    
    def analyze_with_transformers(self, prompt):
        """Analyze using Transformers model"""
        
        try:
            print("🤖 Processing with Transformers...")
            
            # Generate response
            result = self.llm(
                prompt,
                max_length=len(prompt.split()) + 300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=50256  # GPT-2 pad token
            )
            
            # Extract generated text
            generated_text = result[0]['generated_text']
            analysis = generated_text[len(prompt):].strip()
            
            print("✅ Transformers analysis complete")
            return analysis
            
        except Exception as e:
            print(f"❌ Transformers error: {e}")
            return "Error: Could not complete Transformers analysis"
    
    def analyze_with_ollama(self, prompt):
        """Analyze using Ollama model"""
        
        try:
            print("🦙 Processing with Ollama...")
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 500
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get("response", "").strip()
                print("✅ Ollama analysis complete")
                return analysis
            else:
                return f"Error: Ollama API returned status {response.status_code}"
                
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return "Error: Could not complete Ollama analysis"
    
    def structure_llm_output(self, llm_response, case_data):
        """Structure LLM output into organized result"""
        
        # Extract key metrics from LLM response
        deception_likelihood = "MEDIUM"  # Default
        if "high" in llm_response.lower() and ("deception" in llm_response.lower() or "likely" in llm_response.lower()):
            deception_likelihood = "HIGH"
        elif "low" in llm_response.lower() and ("deception" in llm_response.lower() or "unlikely" in llm_response.lower()):
            deception_likelihood = "LOW"
        
        # Count subject statements
        subject_statements = [msg for msg in case_data["conversation_transcript"] if msg["speaker"] == "Subject"]
        
        # Create structured result
        result = {
            "case_information": case_data["case_information"],
            "llm_analysis": {
                "model_used": f"{self.llm_type}",
                "analysis_timestamp": datetime.now().isoformat(),
                "raw_llm_response": llm_response,
                "structured_assessment": {
                    "deception_likelihood": deception_likelihood,
                    "total_subject_statements": len(subject_statements),
                    "analysis_confidence": "High" if len(llm_response) > 200 else "Medium"
                }
            },
            "detailed_findings": {
                "llm_reasoning": llm_response,
                "key_insights": self.extract_key_insights(llm_response),
                "recommendations": self.extract_recommendations(llm_response)
            }
        }
        
        return result
    
    def extract_key_insights(self, llm_response):
        """Extract key insights from LLM response"""
        
        insights = []
        
        # Look for common deception indicators in LLM response
        if "uncertain" in llm_response.lower() or "uncertainty" in llm_response.lower():
            insights.append("LLM detected uncertainty markers in subject's responses")
        
        if "contradict" in llm_response.lower():
            insights.append("LLM identified contradictions in statements")
        
        if "defensive" in llm_response.lower():
            insights.append("LLM noted defensive language patterns")
        
        if "memory" in llm_response.lower() and ("gap" in llm_response.lower() or "selective" in llm_response.lower()):
            insights.append("LLM flagged suspicious memory patterns")
        
        if not insights:
            insights.append("LLM completed analysis - see detailed response")
        
        return insights
    
    def extract_recommendations(self, llm_response):
        """Extract recommendations from LLM response"""
        
        recommendations = []
        
        # Look for investigation suggestions
        if "verify" in llm_response.lower():
            recommendations.append("Verify alibi claims independently")
        
        if "evidence" in llm_response.lower():
            recommendations.append("Confront subject with physical evidence")
        
        if "follow" in llm_response.lower() or "interview" in llm_response.lower():
            recommendations.append("Conduct follow-up interviews")
        
        if not recommendations:
            recommendations.append("Continue investigation based on LLM analysis")
        
        return recommendations

# ============================================================================
# 2. SAMPLE DATA AND MAIN EXECUTION
# ============================================================================

def create_sample_case_data():
    """Create sample case data for LLM analysis"""
    
    return {
        "case_information": {
            "case_id": "CASE-2024-0156",
            "case_type": "theft",
            "incident_description": "Laptop stolen from office building",
            "incident_date": "2024-01-15",
            "incident_time": "14:30",
            "location": "Tech Corp Building, 5th Floor, Room 512"
        },
        "established_facts": {
            "timeline": {
                "14:00": "Building security confirms normal access",
                "14:30": "Victim leaves for meeting", 
                "15:45": "Victim returns, laptop missing"
            },
            "physical_evidence": [
                "Security footage shows suspect entering 5th floor at 14:35",
                "Key card access log shows entry to Room 512 at 14:37",
                "Fingerprints found on victim's desk"
            ]
        },
        "interview_metadata": {
            "interview_date": "2024-01-16",
            "interviewer": "Detective Martinez",
            "subject": "Alex Thompson"
        },
        "conversation_transcript": [
            {
                "timestamp": "10:02:15",
                "speaker": "Detective",
                "content": "Where were you between 2 PM and 4 PM yesterday?"
            },
            {
                "timestamp": "10:02:32",
                "speaker": "Subject", 
                "content": "I was working on network maintenance. I think I was mostly in the server room on the 3rd floor."
            },
            {
                "timestamp": "10:03:15",
                "speaker": "Detective",
                "content": "We have key card records showing access to the 5th floor at 2:37 PM. Can you explain that?"
            },
            {
                "timestamp": "10:03:28",
                "speaker": "Subject",
                "content": "Oh, the 5th floor? I... I might have gone up there briefly. Sometimes I check network connections on different floors. But I wasn't there long."
            },
            {
                "timestamp": "10:04:10",
                "speaker": "Detective",
                "content": "What were you doing in Room 512 specifically?"
            },
            {
                "timestamp": "10:04:18",
                "speaker": "Subject",
                "content": "I honestly don't remember being in that specific room. I mean, I might have poked my head in to check a network port, but I definitely didn't touch anything."
            },
            {
                "timestamp": "10:04:35",
                "speaker": "Detective",
                "content": "A laptop was stolen from that room. Do you know anything about that?"
            },
            {
                "timestamp": "10:04:43",
                "speaker": "Subject",
                "content": "A laptop was stolen? That's terrible! But I had nothing to do with that. I would never steal anything, especially not from someone I work with."
            },
            {
                "timestamp": "10:05:12",
                "speaker": "Subject",
                "content": "I wasn't nervous. I was just focused on my work. Maybe I looked concentrated, but definitely not nervous. Who said I looked nervous?"
            },
            {
                "timestamp": "10:06:08",
                "speaker": "Subject",
                "content": "Look, I'm trying to be helpful here, but yesterday was busy and I had a lot on my mind. My contract is ending this week, so I've been stressed about finding new work. Maybe I'm not remembering everything perfectly."
            }
        ]
    }

def main():
    """Main execution function"""
    
    print("🎯 SIMPLE LLM-BASED LIE DETECTION DEMONSTRATION")
    print("=" * 70)
    
    # Create sample data
    print("📝 Creating sample case data...")
    case_data = create_sample_case_data()
    
    # Save input file
    input_file = "llm_input_case.json"
    with open(input_file, 'w') as f:
        json.dump(case_data, f, indent=2)
    print(f"✅ Input saved: {input_file}")
    
    # Choose LLM type (change this to test different models)
    llm_type = "transformers"  # or "ollama"
    
    print(f"\n🧠 Initializing {llm_type.upper()} LLM...")
    
    try:
        # Initialize LLM detector
        detector = SimpleLLMLieDetector(llm_type=llm_type)
        
        # Run LLM analysis
        print("\n🔍 Running LLM analysis...")
        results = detector.analyze_conversation(case_data)
        
        # Save output file
        output_file = f"llm_analysis_output_{llm_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📤 Output saved: {output_file}")
        
        # Display results
        print("\n" + "=" * 70)
        print("📋 LLM ANALYSIS RESULTS")
        print("=" * 70)
        
        llm_analysis = results["llm_analysis"]
        print(f"🤖 Model Used: {llm_analysis['model_used'].upper()}")
        print(f"📊 Deception Likelihood: {llm_analysis['structured_assessment']['deception_likelihood']}")
        print(f"🎯 Analysis Confidence: {llm_analysis['structured_assessment']['analysis_confidence']}")
        
        print(f"\n🧠 LLM REASONING:")
        print("-" * 50)
        print(llm_analysis['raw_llm_response'])
        
        print(f"\n💡 KEY INSIGHTS:")
        for insight in results["detailed_findings"]["key_insights"]:
            print(f"   • {insight}")
        
        print(f"\n📋 RECOMMENDATIONS:")
        for rec in results["detailed_findings"]["recommendations"]:
            print(f"   • {rec}")
        
        print("\n" + "=" * 70)
        print("✅ LLM Analysis Complete!")
        print(f"📁 Files: {input_file} → {output_file}")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have the required LLM setup:")
        print("  - Transformers: pip install transformers torch")
        print("  - Ollama: Download from ollama.ai and run 'ollama serve'")

if __name__ == "__main__":
    main()