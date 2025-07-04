# ============================================================================
# LLM-BASED STRATEGIC FEASIBILITY ASSESSMENT IN OUTDOOR SPORTS (CRICKET)
# Direct LLM analysis for cricket strategy and tactics
# ============================================================================

import json
import requests
from datetime import datetime
from typing import Dict, List, Tuple

# ============================================================================
# 1. SIMPLE LLM CRICKET STRATEGY ANALYZER
# ============================================================================

class CricketStrategyAnalyzer:
    """LLM-based cricket strategy and feasibility assessment"""
    
    def __init__(self, llm_type="transformers"):
        self.llm_type = llm_type
        print(f"🏏 Initializing {llm_type.upper()} Cricket Strategy Analyzer...")
        
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
            print("📥 Loading Transformers model for cricket analysis...")
            
            self.llm = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium",
                device=-1
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
    
    def analyze_cricket_strategy(self, match_data):
        """Main analysis function using LLM (Algorithm Steps 1-3)"""
        
        print("🏏 Analyzing cricket strategy with LLM...")
        
        # Step 1: Prompt Construction
        prompt = self.construct_strategy_prompt(match_data)
        
        # Step 2: LLM Inference 
        if self.llm_type == "transformers":
            llm_output = self.analyze_with_transformers(prompt)
        elif self.llm_type == "ollama":
            llm_output = self.analyze_with_ollama(prompt)
        
        # Step 3: Extract Feasibility and Strategy
        feasibility, factors, suggestions = self.extract_strategy_components(llm_output)
        
        # Structure complete result
        analysis_result = {
            "match_context": match_data,
            "llm_analysis": {
                "model_used": self.llm_type,
                "analysis_timestamp": datetime.now().isoformat(),
                "raw_llm_output": llm_output,
                "success_feasibility_score": feasibility,
                "key_influencing_factors": factors,
                "tactical_suggestions": suggestions
            },
            "strategic_assessment": {
                "overall_recommendation": self.determine_overall_strategy(feasibility, factors),
                "confidence_level": self.assess_confidence(llm_output),
                "priority_actions": self.extract_priority_actions(suggestions)
            }
        }
        
        return analysis_result
    
    def construct_strategy_prompt(self, match_data):
        """Algorithm Step 1: Construct comprehensive strategy prompt"""
        
        # Extract components
        environmental = match_data["environmental_data"]
        ground_history = match_data["ground_history"]
        team_metadata = match_data["team_metadata"]
        match_context = match_data["match_context"]
        
        # Format environmental conditions
        env_text = f"""
ENVIRONMENTAL CONDITIONS:
• Temperature: {environmental['temperature']}°C
• Humidity: {environmental['humidity']}%
• Wind Speed: {environmental['wind_speed']} km/h, Direction: {environmental['wind_direction']}
• Altitude: {environmental['altitude']} meters
• Expected Weather: {environmental['weather_forecast']}
• Pitch Conditions: {environmental['pitch_conditions']}"""
        
        # Format ground history
        history_text = f"""
GROUND HISTORY & STATISTICS:
• Ground: {ground_history['ground_name']}, {ground_history['location']}
• Average First Innings Score: {ground_history['avg_first_innings_score']}
• Average Second Innings Score: {ground_history['avg_second_innings_score']}
• Toss Win Success Rate: {ground_history['toss_win_percentage']}%
• Most Successful Strategy: {ground_history['most_successful_strategy']}
• Key Ground Characteristics: {', '.join(ground_history['ground_characteristics'])}"""
        
        # Format team information
        team_text = f"""
TEAM PROFILE:
• Team: {team_metadata['team_name']} vs {team_metadata['opponent_team']}
• Home Region: {team_metadata['home_climate_region']}
• Climate Adaptability: {team_metadata['climate_adaptability']}/10
• Recent Form: {team_metadata['recent_form']} (last 5 matches)
• Key Players: {', '.join(team_metadata['key_players'])}
• Squad Strengths: {', '.join(team_metadata['squad_strengths'])}
• Squad Weaknesses: {', '.join(team_metadata['squad_weaknesses'])}"""
        
        # Format match context
        context_text = f"""
MATCH CONTEXT:
• Format: {match_context['format']}
• Tournament: {match_context['tournament']}
• Match Importance: {match_context['importance_level']}
• Expected Crowd: {match_context['crowd_support']}
• Media Pressure: {match_context['media_pressure']}"""
        
        # Create comprehensive prompt
        prompt = f"""You are a world-class cricket strategist and analyst with 20 years of experience in professional cricket. You have successfully guided teams to victory in various international tournaments and conditions.

STRATEGIC ANALYSIS REQUEST:
Analyze the following cricket match scenario and provide comprehensive strategic recommendations.

{env_text}

{history_text}

{team_text}

{context_text}

ANALYSIS REQUIREMENTS:
Based on the above information, provide a detailed strategic assessment covering:

1. SUCCESS FEASIBILITY ASSESSMENT:
   - Overall likelihood of success (High/Moderate/Low)
   - Confidence level in assessment
   - Key risk factors

2. KEY INFLUENCING FACTORS:
   - Environmental factors impact on gameplay
   - Ground characteristics advantages/disadvantages  
   - Team strengths/weaknesses in these conditions
   - Historical patterns and trends

3. TACTICAL SUGGESTIONS:
   - Optimal team selection and batting order
   - Bowling strategy and field placements
   - Toss decision recommendations
   - In-match tactical adjustments
   - Training focus areas before the match

4. CONTINGENCY PLANNING:
   - Alternative strategies for different scenarios
   - Adaptation strategies if conditions change
   - Key decision points during the match

Provide specific, actionable insights based on the match context and your cricket expertise.

STRATEGIC ANALYSIS:"""

        return prompt
    
    def analyze_with_transformers(self, prompt):
        """Algorithm Step 2: LLM Inference using Transformers"""
        
        try:
            print("🤖 Processing cricket strategy with Transformers...")
            
            result = self.llm(
                prompt,
                max_length=len(prompt.split()) + 400,
                temperature=0.6,
                do_sample=True,
                pad_token_id=50256
            )
            
            generated_text = result[0]['generated_text']
            analysis = generated_text[len(prompt):].strip()
            
            print("✅ Transformers cricket analysis complete")
            return analysis
            
        except Exception as e:
            print(f"❌ Transformers error: {e}")
            return "Error: Could not complete Transformers analysis"
    
    def analyze_with_ollama(self, prompt):
        """Algorithm Step 2: LLM Inference using Ollama"""
        
        try:
            print("🦙 Processing cricket strategy with Ollama...")
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "num_predict": 600
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get("response", "").strip()
                print("✅ Ollama cricket analysis complete")
                return analysis
            else:
                return f"Error: Ollama API returned status {response.status_code}"
                
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return "Error: Could not complete Ollama analysis"
    
    def extract_strategy_components(self, llm_output):
        """Algorithm Step 3: Extract feasibility, factors, and suggestions"""
        
        # Extract feasibility score
        feasibility = "Moderate"  # Default
        if "high" in llm_output.lower() and any(word in llm_output.lower() for word in ["success", "feasibility", "likelihood"]):
            feasibility = "High"
        elif "low" in llm_output.lower() and any(word in llm_output.lower() for word in ["success", "feasibility", "likelihood"]):
            feasibility = "Low"
        
        # Extract key factors
        factors = []
        
        # Environmental factors
        if "temperature" in llm_output.lower() or "heat" in llm_output.lower():
            factors.append("Temperature and heat conditions")
        if "humidity" in llm_output.lower():
            factors.append("Humidity levels affecting ball swing")
        if "wind" in llm_output.lower():
            factors.append("Wind conditions impacting gameplay")
        if "pitch" in llm_output.lower():
            factors.append("Pitch conditions and behavior")
        
        # Team factors
        if "adaptability" in llm_output.lower() or "adapt" in llm_output.lower():
            factors.append("Team climate adaptability")
        if "experience" in llm_output.lower():
            factors.append("Player experience in similar conditions")
        if "form" in llm_output.lower():
            factors.append("Recent team form and momentum")
        
        # Tactical factors
        if "bowling" in llm_output.lower():
            factors.append("Bowling strategy effectiveness")
        if "batting" in llm_output.lower():
            factors.append("Batting approach and order")
        if "field" in llm_output.lower():
            factors.append("Field placement strategy")
        
        if not factors:
            factors.append("Comprehensive match conditions analysis")
        
        # Extract tactical suggestions
        suggestions = []
        
        if "toss" in llm_output.lower():
            suggestions.append("Strategic toss decision based on conditions")
        if "bowling" in llm_output.lower() and ("first" in llm_output.lower() or "pace" in llm_output.lower()):
            suggestions.append("Optimize bowling order and pace strategy")
        if "batting" in llm_output.lower() and ("order" in llm_output.lower() or "lineup" in llm_output.lower()):
            suggestions.append("Adjust batting order for conditions")
        if "field" in llm_output.lower() and "placement" in llm_output.lower():
            suggestions.append("Tactical field placement adjustments")
        if "training" in llm_output.lower() or "practice" in llm_output.lower():
            suggestions.append("Targeted pre-match training focus")
        if "selection" in llm_output.lower() or "team" in llm_output.lower():
            suggestions.append("Strategic team selection optimization")
        
        if not suggestions:
            suggestions.append("Comprehensive tactical approach based on analysis")
        
        return feasibility, factors, suggestions
    
    def determine_overall_strategy(self, feasibility, factors):
        """Determine overall strategic recommendation"""
        
        if feasibility == "High":
            return "Aggressive strategy recommended - capitalize on favorable conditions"
        elif feasibility == "Low":
            return "Conservative strategy recommended - minimize risks and adapt cautiously"
        else:
            return "Balanced strategy recommended - adapt tactically based on match flow"
    
    def assess_confidence(self, llm_output):
        """Assess confidence level in LLM analysis"""
        
        if len(llm_output) > 300 and any(word in llm_output.lower() for word in ["recommend", "suggest", "strategy"]):
            return "High"
        elif len(llm_output) > 150:
            return "Medium"
        else:
            return "Low"
    
    def extract_priority_actions(self, suggestions):
        """Extract priority actions from suggestions"""
        
        priorities = []
        
        for suggestion in suggestions:
            if "toss" in suggestion.lower():
                priorities.append("1. Make optimal toss decision")
            elif "selection" in suggestion.lower() or "team" in suggestion.lower():
                priorities.append("2. Finalize team selection")
            elif "training" in suggestion.lower():
                priorities.append("3. Focus pre-match training")
            elif "bowling" in suggestion.lower():
                priorities.append("4. Plan bowling strategy")
            elif "batting" in suggestion.lower():
                priorities.append("5. Set batting approach")
        
        if not priorities:
            priorities = ["1. Review all strategic elements", "2. Adapt tactics to conditions"]
        
        return priorities

# ============================================================================
# 2. SAMPLE DATA AND MAIN EXECUTION
# ============================================================================

def create_sample_cricket_data():
    """Create comprehensive sample cricket match data"""
    
    return {
        "match_context": {
            "format": "T20 International",
            "tournament": "ICC T20 World Cup 2024",
            "importance_level": "Semi-Final",
            "crowd_support": "High - Home advantage",
            "media_pressure": "Very High"
        },
        "environmental_data": {
            "temperature": 35,
            "humidity": 68,
            "wind_speed": 15,
            "wind_direction": "South-West",
            "altitude": 1200,
            "weather_forecast": "Partly cloudy, possible evening showers",
            "pitch_conditions": "Dry, expected to favor spinners in second innings"
        },
        "ground_history": {
            "ground_name": "M. Chinnaswamy Stadium",
            "location": "Bangalore, India",
            "avg_first_innings_score": 168,
            "avg_second_innings_score": 152,
            "toss_win_percentage": 65,
            "most_successful_strategy": "Bowl first, chase with spinners",
            "ground_characteristics": [
                "Short boundaries (60m square)",
                "Fast outfield",
                "Spin-friendly in second innings",
                "Dew factor in evening matches"
            ],
            "recent_matches": [
                {"team1": "India", "team2": "Australia", "score1": 180, "score2": 165, "winner": "India"},
                {"team1": "England", "team2": "South Africa", "score1": 145, "score2": 149, "winner": "South Africa"},
                {"team1": "Pakistan", "team2": "New Zealand", "score1": 175, "score2": 170, "winner": "Pakistan"}
            ]
        },
        "team_metadata": {
            "team_name": "India",
            "opponent_team": "Australia", 
            "home_climate_region": "Tropical/Subtropical",
            "climate_adaptability": 9,
            "recent_form": "W-W-L-W-W",
            "key_players": [
                "Virat Kohli (Batsman)",
                "Jasprit Bumrah (Fast Bowler)", 
                "Ravindra Jadeja (All-rounder)",
                "Rohit Sharma (Captain/Batsman)"
            ],
            "squad_strengths": [
                "Strong spin bowling attack",
                "Experienced batting lineup",
                "Home ground advantage",
                "Excellent fielding unit",
                "Depth in all-rounders"
            ],
            "squad_weaknesses": [
                "Pace bowling in death overs",
                "Middle-order instability",
                "Pressure handling in knockouts"
            ],
            "opponent_analysis": {
                "opponent_strengths": ["Aggressive batting", "Quality pace attack", "Big match experience"],
                "opponent_weaknesses": ["Spin bowling struggle", "Middle overs batting", "Subcontinent conditions"]
            }
        },
        "historical_context": {
            "head_to_head_record": "India 12 - 8 Australia (Last 20 T20Is)",
            "last_encounter": {
                "date": "2024-01-15",
                "venue": "Melbourne",
                "result": "Australia won by 7 runs",
                "key_factors": ["Australian pace bowling", "Indian middle-order collapse"]
            },
            "tournament_performance": {
                "india_progress": "Group: 4/4 wins, Quarter-final: Won vs England",
                "australia_progress": "Group: 3/4 wins, Quarter-final: Won vs South Africa"
            }
        }
    }

def main():
    """Main execution function"""
    
    print("🏏 CRICKET STRATEGIC FEASIBILITY ASSESSMENT")
    print("=" * 70)
    
    # Create sample match data
    print("📊 Creating sample cricket match data...")
    match_data = create_sample_cricket_data()
    
    # Save input file
    input_file = "cricket_match_input.json"
    with open(input_file, 'w') as f:
        json.dump(match_data, f, indent=2)
    print(f"✅ Input saved: {input_file}")
    
    # Choose LLM type
    llm_type = "transformers"  # or "ollama"
    
    print(f"\n🧠 Initializing {llm_type.upper()} Cricket Analyzer...")
    
    try:
        # Initialize analyzer
        analyzer = CricketStrategyAnalyzer(llm_type=llm_type)
        
        # Run LLM analysis
        print("\n🏏 Running cricket strategy analysis...")
        results = analyzer.analyze_cricket_strategy(match_data)
        
        # Save output file
        output_file = f"cricket_strategy_output_{llm_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📤 Output saved: {output_file}")
        
        # Display results
        print("\n" + "=" * 70)
        print("🏏 CRICKET STRATEGY ANALYSIS RESULTS")
        print("=" * 70)
        
        llm_analysis = results["llm_analysis"]
        strategic = results["strategic_assessment"]
        
        print(f"🤖 Model Used: {llm_analysis['model_used'].upper()}")
        print(f"🎯 Success Feasibility: {llm_analysis['success_feasibility_score']}")
        print(f"📊 Confidence Level: {strategic['confidence_level']}")
        print(f"💡 Overall Strategy: {strategic['overall_recommendation']}")
        
        print(f"\n🔍 KEY INFLUENCING FACTORS:")
        for factor in llm_analysis['key_influencing_factors']:
            print(f"   • {factor}")
        
        print(f"\n⚡ TACTICAL SUGGESTIONS:")
        for suggestion in llm_analysis['tactical_suggestions']:
            print(f"   • {suggestion}")
        
        print(f"\n📋 PRIORITY ACTIONS:")
        for action in strategic['priority_actions']:
            print(f"   {action}")
        
        print(f"\n🧠 LLM STRATEGIC REASONING:")
        print("-" * 50)
        print(llm_analysis['raw_llm_output'][:500] + "..." if len(llm_analysis['raw_llm_output']) > 500 else llm_analysis['raw_llm_output'])
        
        print("\n" + "=" * 70)
        print("✅ Cricket Strategy Analysis Complete!")
        print(f"📁 Files: {input_file} → {output_file}")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have the required LLM setup:")
        print("  - Transformers: pip install transformers torch")
        print("  - Ollama: Download from ollama.ai and run 'ollama serve'")

if __name__ == "__main__":
    main()