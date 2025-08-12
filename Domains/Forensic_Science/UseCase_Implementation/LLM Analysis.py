# ============================================================================
# LLM-BASED DECEPTION DETECTION SYSTEM
# Professional implementation using real language models for forensic analysis
# ============================================================================

import json
import requests
from datetime import datetime
from typing import Dict, List, Optional

# ============================================================================
# 1. LLM DECEPTION DETECTION ENGINE
# ============================================================================

class ForensicLLMAnalyzer:
    """Professional LLM-based deception detection system for interrogation analysis"""
    
    def __init__(self, llm_type="transformers"):
        self.llm_type = llm_type
        self.model_ready = False
        
        print(f"Initializing {llm_type.upper()} Language Model for Forensic Analysis...")
        
        if llm_type == "transformers":
            self.setup_transformers()
        elif llm_type == "ollama":
            self.setup_ollama()
        else:
            raise ValueError("Unsupported LLM type. Use 'transformers' or 'ollama'")
    
    def setup_transformers(self):
        """Initialize Hugging Face Transformers model"""
        try:
            from transformers import pipeline
            print("Loading Transformers model for text analysis...")
            
            # Initialize text generation pipeline for analytical reasoning
            self.llm = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium",
                device=-1,  # CPU execution
                return_full_text=False
            )
            
            self.model_ready = True
            print("Transformers model loaded successfully")
            
        except ImportError:
            print("ERROR: Transformers library not found")
            print("Install with: pip install transformers torch")
            raise
        except Exception as e:
            print(f"ERROR: Failed to load Transformers model: {e}")
            raise
    
    def setup_ollama(self):
        """Initialize Ollama LLM connection"""
        try:
            print("Connecting to Ollama server...")
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                if models:
                    # Select first available model
                    self.model_name = models[0]['name']
                    self.model_ready = True
                    print(f"Connected to Ollama model: {self.model_name}")
                else:
                    raise ConnectionError("No Ollama models available. Install with: ollama pull llama2")
            else:
                raise ConnectionError("Ollama server not responding")
                
        except requests.exceptions.RequestException:
            print("ERROR: Cannot connect to Ollama server")
            print("Start Ollama with: ollama serve")
            raise
    
    def analyze_interrogation(self, case_data: Dict) -> Dict:
        """
        Main analysis function for interrogation transcripts
        
        Args:
            case_data: Dictionary containing case information and conversation transcript
            
        Returns:
            Structured analysis results from LLM processing
        """
        
        if not self.model_ready:
            raise RuntimeError("LLM model not properly initialized")
        
        print("Analyzing interrogation transcript...")
        
        # Create analysis prompt
        analysis_prompt = self._create_forensic_prompt(case_data)
        
        # Get LLM analysis
        if self.llm_type == "transformers":
            llm_response = self._analyze_with_transformers(analysis_prompt)
        elif self.llm_type == "ollama":
            llm_response = self._analyze_with_ollama(analysis_prompt)
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")
        
        # Structure and validate output
        analysis_result = self._structure_analysis_output(llm_response, case_data)
        
        return analysis_result
    
    def _create_forensic_prompt(self, case_data: Dict) -> str:
        """Create structured prompt for forensic analysis"""
        
        case_info = case_data["case_information"]
        conversation = case_data["conversation_transcript"]
        facts = case_data["established_facts"]
        
        # Format conversation transcript
        transcript_text = ""
        for entry in conversation:
            speaker_label = "SUSPECT" if entry["speaker"] == "Subject" else "INVESTIGATOR"
            transcript_text += f"{speaker_label}: {entry['content']}\n"
        
        # Build comprehensive prompt
        prompt = f"""You are a forensic psychologist and expert in deception detection through linguistic analysis.

CASE DETAILS:
Type: {case_info['case_type'].title()}
Description: {case_info['incident_description']}
Date/Time: {case_info['incident_date']} at {case_info['incident_time']}
Location: {case_info['location']}

ESTABLISHED EVIDENCE:
Timeline: {json.dumps(facts.get('timeline', {}), indent=2)}
Physical Evidence: {', '.join(facts.get('physical_evidence', []))}

INTERROGATION TRANSCRIPT:
{transcript_text}

FORENSIC ANALYSIS REQUIRED:
Conduct a comprehensive deception analysis focusing on:

1. LINGUISTIC INDICATORS: Speech patterns suggesting deception or truthfulness
2. CONSISTENCY ANALYSIS: Internal contradictions or alignment with known facts
3. PSYCHOLOGICAL MARKERS: Behavioral indicators of stress, evasion, or fabrication
4. CREDIBILITY ASSESSMENT: Overall likelihood of truthful testimony

Provide detailed reasoning with specific examples from the transcript. Rate deception likelihood as HIGH, MEDIUM, or LOW with supporting evidence.

PROFESSIONAL ANALYSIS:"""

        return prompt
    
    def _analyze_with_transformers(self, prompt: str) -> str:
        """Process analysis using Transformers model"""
        
        try:
            print("Processing with Transformers language model...")
            
            # Generate analytical response
            result = self.llm(
                prompt,
                max_new_tokens=400,
                temperature=0.3,  # Low temperature for analytical consistency
                do_sample=True,
                pad_token_id=50256
            )
            
            # Extract generated analysis
            if isinstance(result, list) and len(result) > 0:
                analysis = result[0].get('generated_text', '').strip()
            else:
                analysis = str(result).strip()
            
            print("Transformers analysis completed")
            return analysis
            
        except Exception as e:
            error_msg = f"Transformers processing error: {e}"
            print(f"ERROR: {error_msg}")
            return error_msg
    
    def _analyze_with_ollama(self, prompt: str) -> str:
        """Process analysis using Ollama model"""
        
        try:
            print("Processing with Ollama language model...")
            
            request_payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Low temperature for consistent analysis
                    "top_p": 0.95,
                    "num_predict": 600,
                    "stop": ["ANALYSIS COMPLETE"]
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=request_payload,
                timeout=120  # Extended timeout for complex analysis
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get("response", "").strip()
                print("Ollama analysis completed")
                return analysis
            else:
                error_msg = f"Ollama API error: HTTP {response.status_code}"
                print(f"ERROR: {error_msg}")
                return error_msg
                
        except Exception as e:
            error_msg = f"Ollama processing error: {e}"
            print(f"ERROR: {error_msg}")
            return error_msg
    
    def _structure_analysis_output(self, llm_response: str, case_data: Dict) -> Dict:
        """Structure LLM output into standardized forensic report"""
        
        # Determine deception likelihood from LLM response
        deception_assessment = self._extract_deception_likelihood(llm_response)
        
        # Count relevant metrics
        subject_statements = [msg for msg in case_data["conversation_transcript"] 
                            if msg["speaker"] == "Subject"]
        
        # Extract analytical components
        key_findings = self._extract_key_findings(llm_response)
        recommendations = self._extract_investigative_recommendations(llm_response)
        
        # Build structured result
        analysis_result = {
            "case_metadata": {
                "case_id": case_data["case_information"].get("case_id", "UNKNOWN"),
                "analysis_date": datetime.now().isoformat(),
                "analyst_type": f"LLM ({self.llm_type})",
                "model_version": getattr(self, 'model_name', 'DialoGPT-medium')
            },
            "forensic_assessment": {
                "deception_likelihood": deception_assessment,
                "confidence_level": self._assess_analysis_confidence(llm_response),
                "statement_count": len(subject_statements),
                "analysis_quality": "High" if len(llm_response) > 300 else "Standard"
            },
            "detailed_analysis": {
                "llm_full_response": llm_response,
                "key_findings": key_findings,
                "investigative_recommendations": recommendations,
                "risk_factors": self._identify_risk_factors(llm_response)
            },
            "case_context": case_data["case_information"]
        }
        
        return analysis_result
    
    def _extract_deception_likelihood(self, llm_response: str) -> str:
        """Extract deception likelihood assessment from LLM response"""
        
        response_lower = llm_response.lower()
        
        # Look for explicit likelihood statements
        if any(phrase in response_lower for phrase in ["high likelihood", "highly likely", "strong indicators"]):
            return "HIGH"
        elif any(phrase in response_lower for phrase in ["low likelihood", "unlikely", "minimal indicators"]):
            return "LOW"
        elif any(phrase in response_lower for phrase in ["medium", "moderate", "some indicators"]):
            return "MEDIUM"
        
        # Fallback assessment based on content analysis
        deception_indicators = ["contradict", "inconsisten", "evasive", "defensive", "uncertain"]
        truthfulness_indicators = ["consistent", "direct", "cooperative", "detailed", "logical"]
        
        deception_score = sum(1 for indicator in deception_indicators if indicator in response_lower)
        truthfulness_score = sum(1 for indicator in truthfulness_indicators if indicator in response_lower)
        
        if deception_score > truthfulness_score + 1:
            return "HIGH"
        elif truthfulness_score > deception_score + 1:
            return "LOW"
        else:
            return "MEDIUM"
    
    def _assess_analysis_confidence(self, llm_response: str) -> str:
        """Assess confidence level of the analysis"""
        
        if len(llm_response) > 400 and any(word in llm_response.lower() 
                                         for word in ["evidence", "specific", "clear", "obvious"]):
            return "High"
        elif len(llm_response) > 200:
            return "Medium"
        else:
            return "Low"
    
    def _extract_key_findings(self, llm_response: str) -> List[str]:
        """Extract key findings from LLM analysis"""
        
        findings = []
        response_lower = llm_response.lower()
        
        # Detect specific analytical observations
        if "memory" in response_lower and ("selective" in response_lower or "gaps" in response_lower):
            findings.append("Selective memory patterns detected in subject responses")
        
        if "contradict" in response_lower:
            findings.append("Internal contradictions identified in testimony")
        
        if "defensive" in response_lower or "evasive" in response_lower:
            findings.append("Defensive or evasive communication patterns observed")
        
        if "stress" in response_lower or "nervous" in response_lower:
            findings.append("Indicators of psychological stress during questioning")
        
        if "timeline" in response_lower and "inconsisten" in response_lower:
            findings.append("Timeline inconsistencies noted in subject account")
        
        if "detail" in response_lower and ("lack" in response_lower or "vague" in response_lower):
            findings.append("Insufficient detail or vague responses to direct questions")
        
        # Ensure at least one finding is recorded
        if not findings:
            findings.append("Analysis completed - refer to detailed response for specific observations")
        
        return findings
    
    def _extract_investigative_recommendations(self, llm_response: str) -> List[str]:
        """Extract investigative recommendations from analysis"""
        
        recommendations = []
        response_lower = llm_response.lower()
        
        if "verify" in response_lower or "corroborate" in response_lower:
            recommendations.append("Independently verify subject's alibi and timeline claims")
        
        if "confront" in response_lower or "evidence" in response_lower:
            recommendations.append("Present physical evidence to subject for direct response")
        
        if "follow" in response_lower and "interview" in response_lower:
            recommendations.append("Conduct follow-up interviews with additional questioning")
        
        if "witness" in response_lower:
            recommendations.append("Interview potential witnesses to corroborate or contradict statements")
        
        if "psychological" in response_lower or "expert" in response_lower:
            recommendations.append("Consider professional psychological evaluation")
        
        # Default recommendation if none detected
        if not recommendations:
            recommendations.append("Continue investigation based on analytical findings")
        
        return recommendations
    
    def _identify_risk_factors(self, llm_response: str) -> List[str]:
        """Identify risk factors in the analysis"""
        
        risk_factors = []
        response_lower = llm_response.lower()
        
        if "flee" in response_lower or "flight" in response_lower:
            risk_factors.append("Potential flight risk identified")
        
        if "destroy" in response_lower or "tamper" in response_lower:
            risk_factors.append("Evidence tampering concerns")
        
        if "repeat" in response_lower or "pattern" in response_lower:
            risk_factors.append("Possible repeat offender patterns")
        
        if len(risk_factors) == 0:
            risk_factors.append("Standard investigative protocols recommended")
        
        return risk_factors

# ============================================================================
# 2. SAMPLE DATA GENERATION AND UTILITIES
# ============================================================================

def generate_sample_case() -> Dict:
    """Generate realistic sample case data for system demonstration"""
    
    return {
        "case_information": {
            "case_id": "INV-2024-0087",
            "case_type": "corporate_theft",
            "incident_description": "High-value laptop containing proprietary software stolen from secure office",
            "incident_date": "2024-01-15",
            "incident_time": "14:30",
            "location": "TechCorp Headquarters, Fifth Floor, Executive Suite 512"
        },
        "established_facts": {
            "timeline": {
                "14:00": "Security system shows normal building access patterns",
                "14:30": "Victim (Sarah Chen) leaves for scheduled client meeting", 
                "14:35": "Key card access logged for suspect at fifth floor entrance",
                "14:37": "Door access to Suite 512 recorded",
                "15:45": "Victim returns from meeting, discovers laptop missing",
                "16:00": "Security notified, building lockdown initiated"
            },
            "physical_evidence": [
                "Security camera footage shows suspect entering fifth floor at 14:35",
                "Electronic key card logs confirm access to Suite 512 at 14:37",
                "Latent fingerprints recovered from victim's desk surface",
                "Network logs show laptop was not remotely accessed during timeframe"
            ],
            "witness_statements": [
                "Security guard confirms normal entry procedures for suspect",
                "Colleague reports seeing suspect near elevators around 14:40"
            ]
        },
        "interview_metadata": {
            "interview_date": "2024-01-16",
            "interview_time": "10:00",
            "interviewer": "Detective Maria Santos",
            "subject": "Alex Thompson (IT Technician)",
            "location": "Police Station Interview Room B"
        },
        "conversation_transcript": [
            {
                "timestamp": "10:02:15",
                "speaker": "Detective",
                "content": "Can you account for your whereabouts between 2:00 PM and 4:00 PM yesterday afternoon?"
            },
            {
                "timestamp": "10:02:32",
                "speaker": "Subject", 
                "content": "Yesterday afternoon? I was handling routine network maintenance tasks. I believe I spent most of my time in the server room on the third floor working on system updates."
            },
            {
                "timestamp": "10:03:15",
                "speaker": "Detective",
                "content": "Our records show your key card was used to access the fifth floor at 2:37 PM. How do you explain that access?"
            },
            {
                "timestamp": "10:03:28",
                "speaker": "Subject",
                "content": "The fifth floor? Oh, right. I might have gone up there briefly. Sometimes network issues require checking connections on multiple floors. I don't stay long when I do those checks."
            },
            {
                "timestamp": "10:04:10",
                "speaker": "Detective",
                "content": "Specifically, you accessed Suite 512. What business did you have in that particular office?"
            },
            {
                "timestamp": "10:04:18",
                "speaker": "Subject",
                "content": "Suite 512? I honestly can't recall being in that specific room. If I was there, it would have been just to check a network connection point. I definitely didn't handle any equipment or personal items."
            },
            {
                "timestamp": "10:04:35",
                "speaker": "Detective",
                "content": "A laptop was stolen from Suite 512 during the time you were on that floor. Do you have any knowledge of this theft?"
            },
            {
                "timestamp": "10:04:43",
                "speaker": "Subject",
                "content": "Someone stole a laptop? That's shocking! I had absolutely nothing to do with any theft. I would never steal from colleagues or the company. That goes against everything I believe in."
            },
            {
                "timestamp": "10:05:12",
                "speaker": "Subject",
                "content": "I wasn't acting nervous yesterday. I was just focused and maybe a bit stressed about work deadlines. If someone thought I looked nervous, they misread my concentration. Who reported that I seemed nervous?"
            },
            {
                "timestamp": "10:06:08",
                "speaker": "Subject",
                "content": "Look, I'm trying to be completely honest and helpful here. Yesterday was an extremely busy day and I had multiple priorities competing for my attention. My employment contract expires this Friday, so I've been under pressure to complete projects and transition my responsibilities. Perhaps my memory of specific details isn't perfect because of that stress."
            }
        ]
    }

def save_analysis_results(results: Dict, filename: Optional[str] = None) -> str:
    """Save analysis results to JSON file with proper formatting"""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        case_id = results.get("case_metadata", {}).get("case_id", "UNKNOWN")
        filename = f"forensic_analysis_{case_id}_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return filename
    except Exception as e:
        print(f"ERROR: Could not save results to {filename}: {e}")
        return ""

def print_analysis_summary(results: Dict) -> None:
    """Print formatted summary of analysis results"""
    
    print("\n" + "=" * 80)
    print("FORENSIC ANALYSIS SUMMARY REPORT")
    print("=" * 80)
    
    # Case information
    case_meta = results["case_metadata"]
    forensic = results["forensic_assessment"]
    
    print(f"Case ID: {case_meta['case_id']}")
    print(f"Analysis Date: {case_meta['analysis_date'][:19]}")
    print(f"Analyst: {case_meta['analyst_type']}")
    print(f"Model: {case_meta['model_version']}")
    
    print(f"\nFORENSIC ASSESSMENT:")
    print(f"   Deception Likelihood: {forensic['deception_likelihood']}")
    print(f"   Confidence Level: {forensic['confidence_level']}")
    print(f"   Statements Analyzed: {forensic['statement_count']}")
    print(f"   Analysis Quality: {forensic['analysis_quality']}")
    
    # Key findings
    print(f"\nKEY FINDINGS:")
    for finding in results["detailed_analysis"]["key_findings"]:
        print(f"   • {finding}")
    
    # Recommendations
    print(f"\nINVESTIGATIVE RECOMMENDATIONS:")
    for recommendation in results["detailed_analysis"]["investigative_recommendations"]:
        print(f"   • {recommendation}")
    
    # Risk factors
    print(f"\nRISK ASSESSMENT:")
    for risk in results["detailed_analysis"]["risk_factors"]:
        print(f"   • {risk}")
    
    print("\n" + "=" * 80)

# ============================================================================
# 3. MAIN EXECUTION FRAMEWORK
# ============================================================================

def main():
    """Main execution function for forensic analysis system"""
    
    print("FORENSIC LLM DECEPTION DETECTION SYSTEM")
    print("=" * 80)
    print("Professional interrogation analysis using language model technology")
    
    # Generate sample case data
    print("\nGenerating sample case data...")
    case_data = generate_sample_case()
    
    # Save input case file
    input_filename = "case_input_data.json"
    try:
        with open(input_filename, 'w', encoding='utf-8') as f:
            json.dump(case_data, f, indent=2, ensure_ascii=False)
        print(f"Case data saved: {input_filename}")
    except Exception as e:
        print(f"WARNING: Could not save input file: {e}")
    
    # Configure LLM type (modify as needed for testing)
    llm_type = "transformers"  # Options: "transformers" or "ollama"
    
    print(f"\nInitializing {llm_type.upper()} analysis system...")
    
    try:
        # Initialize forensic analyzer
        analyzer = ForensicLLMAnalyzer(llm_type=llm_type)
        
        # Perform forensic analysis
        print("\nCommencing forensic analysis of interrogation transcript...")
        analysis_results = analyzer.analyze_interrogation(case_data)
        
        # Save analysis results
        output_filename = save_analysis_results(analysis_results)
        if output_filename:
            print(f"Analysis results saved: {output_filename}")
        
        # Display comprehensive summary
        print_analysis_summary(analysis_results)
        
        # Display detailed LLM response
        print("\nDETAILED LLM ANALYSIS:")
        print("-" * 80)
        print(analysis_results["detailed_analysis"]["llm_full_response"])
        print("-" * 80)
        
        print(f"\nAnalysis complete. Files generated:")
        print(f"   Input: {input_filename}")
        if output_filename:
            print(f"   Output: {output_filename}")
        
    except Exception as e:
        print(f"\nERROR: Analysis failed: {e}")
        print("\nSystem Requirements:")
        print("   For Transformers: pip install transformers torch")
        print("   For Ollama: Install from ollama.ai and run 'ollama serve'")
        print("   Ensure sufficient system memory for model loading")

if __name__ == "__main__":
    main()
