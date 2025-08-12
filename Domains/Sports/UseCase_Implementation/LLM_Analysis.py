# ============================================================================
# CRICKET STRATEGIC FEASIBILITY ASSESSMENT SYSTEM
# Professional LLM-based analysis for cricket strategy and tactical planning
# ============================================================================

import json
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ============================================================================
# 1. CRICKET STRATEGY ANALYSIS ENGINE
# ============================================================================

class CricketStrategicAnalyzer:
    """Professional cricket strategy and feasibility assessment using language models"""
    
    def __init__(self, llm_type="transformers"):
        self.llm_type = llm_type
        self.model_ready = False
        
        print(f"Initializing {llm_type.upper()} Cricket Strategic Analysis System...")
        
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
            print("Loading Transformers model for cricket analysis...")
            
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
    
    def analyze_match_strategy(self, match_data: Dict) -> Dict:
        """
        Main analysis function for cricket strategy assessment
        
        Args:
            match_data: Comprehensive match context and environmental data
            
        Returns:
            Strategic analysis results with feasibility assessment
        """
        
        if not self.model_ready:
            raise RuntimeError("LLM model not properly initialized")
        
        print("Analyzing cricket match strategy...")
        
        # Step 1: Construct strategic analysis prompt
        strategy_prompt = self._construct_strategic_prompt(match_data)
        
        # Step 2: Execute LLM inference
        if self.llm_type == "transformers":
            llm_response = self._analyze_with_transformers(strategy_prompt)
        elif self.llm_type == "ollama":
            llm_response = self._analyze_with_ollama(strategy_prompt)
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")
        
        # Step 3: Extract and structure strategic components
        feasibility_score, key_factors, tactical_recommendations = self._extract_strategic_components(llm_response)
        
        # Compile comprehensive analysis result
        analysis_result = self._compile_strategic_assessment(
            match_data, llm_response, feasibility_score, key_factors, tactical_recommendations
        )
        
        return analysis_result
    
    def _construct_strategic_prompt(self, match_data: Dict) -> str:
        """Construct comprehensive strategic analysis prompt"""
        
        environmental = match_data["environmental_data"]
        ground_stats = match_data["ground_history"]
        team_profile = match_data["team_metadata"]
        match_context = match_data["match_context"]
        
        # Format environmental analysis section
        environmental_section = f"""
ENVIRONMENTAL CONDITIONS ANALYSIS:
• Temperature: {environmental['temperature']}°C (Impact on player performance and ball behavior)
• Humidity: {environmental['humidity']}% (Affects ball swing and player stamina)
• Wind: {environmental['wind_speed']} km/h from {environmental['wind_direction']} (Ball trajectory influence)
• Altitude: {environmental['altitude']}m above sea level (Air density effects)
• Weather Forecast: {environmental['weather_forecast']}
• Pitch Assessment: {environmental['pitch_conditions']}"""
        
        # Format ground statistics section
        ground_section = f"""
GROUND STATISTICAL ANALYSIS:
• Venue: {ground_stats['ground_name']}, {ground_stats['location']}
• Historical Scoring Patterns:
  - First Innings Average: {ground_stats['avg_first_innings_score']} runs
  - Second Innings Average: {ground_stats['avg_second_innings_score']} runs
• Toss Impact: {ground_stats['toss_win_percentage']}% success rate for toss winners
• Optimal Strategy History: {ground_stats['most_successful_strategy']}
• Ground Characteristics: {', '.join(ground_stats['ground_characteristics'])}"""
        
        # Format team profile section
        team_section = f"""
TEAM PROFILE ASSESSMENT:
• Teams: {team_profile['team_name']} vs {team_profile['opponent_team']}
• Climate Adaptation: {team_profile['team_name']} from {team_profile['home_climate_region']}
• Adaptability Rating: {team_profile['climate_adaptability']}/10
• Current Form: {team_profile['recent_form']} (last 5 matches)
• Key Personnel: {', '.join(team_profile['key_players'])}
• Strategic Strengths: {', '.join(team_profile['squad_strengths'])}
• Identified Weaknesses: {', '.join(team_profile['squad_weaknesses'])}"""
        
        # Format match context section
        context_section = f"""
MATCH CONTEXT EVALUATION:
• Format: {match_context['format']}
• Tournament Stage: {match_context['tournament']}
• Stakes: {match_context['importance_level']}
• Crowd Factor: {match_context['crowd_support']}
• Pressure Level: {match_context['media_pressure']}"""
        
        # Construct comprehensive strategic prompt
        prompt = f"""You are a professional cricket strategist with extensive experience in international cricket analysis and team management. Your expertise includes tactical planning, environmental adaptation, and strategic decision-making across all formats of the game.

STRATEGIC ASSESSMENT REQUEST:
Conduct a comprehensive strategic feasibility analysis for the following cricket match scenario.

{environmental_section}

{ground_section}

{team_section}

{context_section}

STRATEGIC ANALYSIS FRAMEWORK:
Please provide a detailed strategic assessment covering the following areas:

1. SUCCESS FEASIBILITY EVALUATION:
   - Overall success probability assessment (High/Moderate/Low)
   - Confidence level in strategic recommendations
   - Critical risk factors and mitigation strategies

2. KEY STRATEGIC FACTORS:
   - Environmental conditions impact on tactical approach
   - Ground characteristics advantages and challenges
   - Team composition optimization for conditions
   - Historical performance patterns and trends

3. TACTICAL IMPLEMENTATION STRATEGY:
   - Optimal team selection and batting order configuration
   - Bowling attack strategy and field placement tactics
   - Toss decision framework and rationale
   - In-match tactical adaptation protocols
   - Pre-match preparation focus areas

4. CONTINGENCY PLANNING:
   - Alternative strategic approaches for varying scenarios
   - Adaptation mechanisms for changing conditions
   - Critical decision points and response strategies

Provide specific, actionable strategic recommendations based on professional cricket analysis principles.

PROFESSIONAL STRATEGIC ANALYSIS:"""

        return prompt
    
    def _analyze_with_transformers(self, prompt: str) -> str:
        """Execute strategic analysis using Transformers model"""
        
        try:
            print("Processing strategic analysis with Transformers...")
            
            result = self.llm(
                prompt,
                max_new_tokens=500,
                temperature=0.4,  # Moderate temperature for balanced creativity and consistency
                do_sample=True,
                pad_token_id=50256
            )
            
            if isinstance(result, list) and len(result) > 0:
                analysis = result[0].get('generated_text', '').strip()
            else:
                analysis = str(result).strip()
            
            print("Transformers strategic analysis completed")
            return analysis
            
        except Exception as e:
            error_msg = f"Transformers processing error: {e}"
            print(f"ERROR: {error_msg}")
            return error_msg
    
    def _analyze_with_ollama(self, prompt: str) -> str:
        """Execute strategic analysis using Ollama model"""
        
        try:
            print("Processing strategic analysis with Ollama...")
            
            request_payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Conservative temperature for strategic consistency
                    "top_p": 0.9,
                    "num_predict": 700,
                    "stop": ["ANALYSIS COMPLETE", "END ANALYSIS"]
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=request_payload,
                timeout=150  # Extended timeout for comprehensive analysis
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get("response", "").strip()
                print("Ollama strategic analysis completed")
                return analysis
            else:
                error_msg = f"Ollama API error: HTTP {response.status_code}"
                print(f"ERROR: {error_msg}")
                return error_msg
                
        except Exception as e:
            error_msg = f"Ollama processing error: {e}"
            print(f"ERROR: {error_msg}")
            return error_msg
    
    def _extract_strategic_components(self, llm_response: str) -> Tuple[str, List[str], List[str]]:
        """Extract strategic components from LLM analysis"""
        
        # Determine feasibility assessment
        feasibility_score = self._assess_success_feasibility(llm_response)
        
        # Extract key influencing factors
        key_factors = self._identify_strategic_factors(llm_response)
        
        # Extract tactical recommendations
        tactical_recommendations = self._extract_tactical_suggestions(llm_response)
        
        return feasibility_score, key_factors, tactical_recommendations
    
    def _assess_success_feasibility(self, llm_response: str) -> str:
        """Assess success feasibility from LLM response"""
        
        response_lower = llm_response.lower()
        
        # Look for explicit feasibility indicators
        high_indicators = ["high success", "highly favorable", "strong advantage", "excellent conditions"]
        low_indicators = ["low success", "challenging conditions", "significant disadvantage", "unfavorable"]
        
        if any(indicator in response_lower for indicator in high_indicators):
            return "High"
        elif any(indicator in response_lower for indicator in low_indicators):
            return "Low"
        elif any(phrase in response_lower for phrase in ["moderate", "balanced", "reasonable"]):
            return "Moderate"
        
        # Analyze strategic language patterns
        positive_terms = ["advantage", "favorable", "strength", "opportunity", "optimal"]
        negative_terms = ["challenge", "difficulty", "weakness", "risk", "concern"]
        
        positive_count = sum(1 for term in positive_terms if term in response_lower)
        negative_count = sum(1 for term in negative_terms if term in response_lower)
        
        if positive_count > negative_count + 1:
            return "High"
        elif negative_count > positive_count + 1:
            return "Low"
        else:
            return "Moderate"
    
    def _identify_strategic_factors(self, llm_response: str) -> List[str]:
        """Identify key strategic factors from analysis"""
        
        factors = []
        response_lower = llm_response.lower()
        
        # Environmental factors
        if any(term in response_lower for term in ["temperature", "heat", "climate"]):
            factors.append("Temperature and climatic conditions impact")
        if "humidity" in response_lower:
            factors.append("Humidity effects on ball movement and player performance")
        if "wind" in response_lower:
            factors.append("Wind conditions affecting ball trajectory")
        if "pitch" in response_lower and any(term in response_lower for term in ["condition", "behavior", "surface"]):
            factors.append("Pitch characteristics and playing surface behavior")
        
        # Team performance factors
        if "adaptation" in response_lower or "acclimatization" in response_lower:
            factors.append("Team adaptation to local conditions")
        if "experience" in response_lower and "conditions" in response_lower:
            factors.append("Player experience in similar environmental conditions")
        if "form" in response_lower or "momentum" in response_lower:
            factors.append("Current team form and performance momentum")
        
        # Tactical factors
        if "bowling" in response_lower and any(term in response_lower for term in ["strategy", "attack", "approach"]):
            factors.append("Bowling strategy and attack configuration")
        if "batting" in response_lower and any(term in response_lower for term in ["order", "approach", "strategy"]):
            factors.append("Batting order and approach optimization")
        if "field" in response_lower and "placement" in response_lower:
            factors.append("Field placement and tactical positioning")
        if "toss" in response_lower:
            factors.append("Toss decision and its strategic implications")
        
        # Ensure minimum factor count
        if len(factors) < 3:
            factors.extend([
                "Match conditions analysis",
                "Team composition considerations",
                "Strategic planning requirements"
            ])
        
        return factors[:6]  # Limit to most relevant factors
    
    def _extract_tactical_suggestions(self, llm_response: str) -> List[str]:
        """Extract tactical suggestions from analysis"""
        
        suggestions = []
        response_lower = llm_response.lower()
        
        # Team selection and composition
        if "selection" in response_lower or "composition" in response_lower:
            suggestions.append("Optimize team selection based on environmental conditions")
        
        # Toss strategy
        if "toss" in response_lower and any(term in response_lower for term in ["decision", "strategy", "choose"]):
            suggestions.append("Strategic toss decision framework implementation")
        
        # Bowling tactics
        if "bowling" in response_lower and any(term in response_lower for term in ["order", "rotation", "strategy"]):
            suggestions.append("Bowling attack rotation and strategic deployment")
        
        # Batting approach
        if "batting" in response_lower and any(term in response_lower for term in ["approach", "strategy", "order"]):
            suggestions.append("Batting order configuration and approach strategy")
        
        # Field placement
        if "field" in response_lower and "placement" in response_lower:
            suggestions.append("Dynamic field placement strategy implementation")
        
        # Training and preparation
        if any(term in response_lower for term in ["training", "preparation", "practice", "conditioning"]):
            suggestions.append("Targeted pre-match preparation and conditioning")
        
        # Adaptation strategies
        if "adapt" in response_lower or "adjust" in response_lower:
            suggestions.append("In-match tactical adaptation protocols")
        
        # Ensure minimum suggestion count
        if len(suggestions) < 3:
            suggestions.extend([
                "Comprehensive tactical planning implementation",
                "Strategic decision-making framework",
                "Performance optimization strategies"
            ])
        
        return suggestions[:6]  # Limit to most actionable suggestions
    
    def _compile_strategic_assessment(self, match_data: Dict, llm_response: str, 
                                    feasibility: str, factors: List[str], 
                                    recommendations: List[str]) -> Dict:
        """Compile comprehensive strategic assessment"""
        
        overall_strategy = self._determine_overall_strategic_approach(feasibility, factors)
        confidence_level = self._assess_analysis_confidence(llm_response)
        priority_actions = self._extract_priority_action_items(recommendations)
        
        assessment = {
            "match_metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "match_format": match_data["match_context"]["format"],
                "tournament": match_data["match_context"]["tournament"],
                "teams": f"{match_data['team_metadata']['team_name']} vs {match_data['team_metadata']['opponent_team']}",
                "venue": f"{match_data['ground_history']['ground_name']}, {match_data['ground_history']['location']}"
            },
            "strategic_analysis": {
                "model_platform": self.llm_type,
                "analysis_timestamp": datetime.now().isoformat(),
                "success_feasibility_score": feasibility,
                "confidence_rating": confidence_level,
                "key_strategic_factors": factors,
                "tactical_recommendations": recommendations,
                "complete_llm_analysis": llm_response
            },
            "strategic_framework": {
                "overall_strategic_approach": overall_strategy,
                "priority_action_items": priority_actions,
                "risk_assessment": self._assess_strategic_risks(llm_response),
                "implementation_timeline": self._generate_implementation_timeline(recommendations)
            },
            "environmental_context": match_data["environmental_data"],
            "team_analysis": match_data["team_metadata"]
        }
        
        return assessment
    
    def _determine_overall_strategic_approach(self, feasibility: str, factors: List[str]) -> str:
        """Determine overall strategic approach recommendation"""
        
        if feasibility == "High":
            return "Aggressive strategic approach - leverage favorable conditions for maximum advantage"
        elif feasibility == "Low":
            return "Conservative strategic approach - minimize risks and focus on damage limitation"
        else:
            return "Balanced strategic approach - adapt tactically based on match progression"
    
    def _assess_analysis_confidence(self, llm_response: str) -> str:
        """Assess confidence level in strategic analysis"""
        
        response_length = len(llm_response)
        strategic_terms = ["strategy", "tactical", "recommend", "approach", "optimal"]
        
        if (response_length > 400 and 
            sum(1 for term in strategic_terms if term in llm_response.lower()) >= 3):
            return "High"
        elif response_length > 200:
            return "Medium"
        else:
            return "Low"
    
    def _extract_priority_action_items(self, recommendations: List[str]) -> List[str]:
        """Extract priority action items from recommendations"""
        
        priority_items = []
        
        for idx, recommendation in enumerate(recommendations, 1):
            if "toss" in recommendation.lower():
                priority_items.append(f"{idx}. Execute optimal toss decision strategy")
            elif "selection" in recommendation.lower() or "team" in recommendation.lower():
                priority_items.append(f"{idx}. Finalize strategic team selection")
            elif "preparation" in recommendation.lower() or "training" in recommendation.lower():
                priority_items.append(f"{idx}. Implement targeted preparation protocols")
            elif "bowling" in recommendation.lower():
                priority_items.append(f"{idx}. Configure bowling strategy framework")
            elif "batting" in recommendation.lower():
                priority_items.append(f"{idx}. Establish batting approach strategy")
            else:
                priority_items.append(f"{idx}. {recommendation}")
        
        return priority_items[:5]  # Limit to top 5 priorities
    
    def _assess_strategic_risks(self, llm_response: str) -> List[str]:
        """Assess strategic risks from analysis"""
        
        risks = []
        response_lower = llm_response.lower()
        
        if any(term in response_lower for term in ["weather", "rain", "storm"]):
            risks.append("Weather disruption potential")
        
        if "pressure" in response_lower or "stress" in response_lower:
            risks.append("High-pressure situation management")
        
        if "adaptation" in response_lower or "unfamiliar" in response_lower:
            risks.append("Environmental adaptation challenges")
        
        if "injury" in response_lower or "fatigue" in response_lower:
            risks.append("Player fitness and injury concerns")
        
        if not risks:
            risks.append("Standard match-related risks")
        
        return risks
    
    def _generate_implementation_timeline(self, recommendations: List[str]) -> Dict[str, List[str]]:
        """Generate implementation timeline for recommendations"""
        
        timeline = {
            "pre_match": [],
            "toss_decision": [],
            "early_match": [],
            "mid_match": []
        }
        
        for rec in recommendations:
            if any(term in rec.lower() for term in ["preparation", "training", "selection"]):
                timeline["pre_match"].append(rec)
            elif "toss" in rec.lower():
                timeline["toss_decision"].append(rec)
            elif any(term in rec.lower() for term in ["opening", "early", "first"]):
                timeline["early_match"].append(rec)
            else:
                timeline["mid_match"].append(rec)
        
        return timeline

# ============================================================================
# 2. SAMPLE DATA GENERATION AND UTILITIES
# ============================================================================

def generate_comprehensive_match_data() -> Dict:
    """Generate comprehensive cricket match data for analysis"""
    
    return {
        "match_context": {
            "format": "T20 International",
            "tournament": "ICC T20 World Cup 2024 Semi-Final",
            "importance_level": "Knockout Stage - Semi-Final",
            "crowd_support": "High intensity - Home advantage expected",
            "media_pressure": "Maximum - Global audience and elimination stakes"
        },
        "environmental_data": {
            "temperature": 35,
            "humidity": 68,
            "wind_speed": 15,
            "wind_direction": "South-West",
            "altitude": 1200,
            "weather_forecast": "Partly cloudy with possible evening precipitation",
            "pitch_conditions": "Dry surface expected to favor spinners in second innings"
        },
        "ground_history": {
            "ground_name": "M. Chinnaswamy Stadium",
            "location": "Bangalore, India",
            "avg_first_innings_score": 168,
            "avg_second_innings_score": 152,
            "toss_win_percentage": 65,
            "most_successful_strategy": "Bowl first, utilize spinners in chase phase",
            "ground_characteristics": [
                "Short square boundaries (60 meters)",
                "Fast outfield with minimal friction",
                "Spin-friendly conditions developing in second innings",
                "Evening dew factor affecting ball grip"
            ],
            "recent_performance_data": [
                {"teams": "India vs Australia", "first_innings": 180, "second_innings": 165, "winner": "India"},
                {"teams": "England vs South Africa", "first_innings": 145, "second_innings": 149, "winner": "South Africa"},
                {"teams": "Pakistan vs New Zealand", "first_innings": 175, "second_innings": 170, "winner": "Pakistan"}
            ]
        },
        "team_metadata": {
            "team_name": "India",
            "opponent_team": "Australia", 
            "home_climate_region": "Tropical/Subtropical monsoon climate",
            "climate_adaptability": 9,
            "recent_form": "Won 4 of last 5 matches (W-W-L-W-W)",
            "key_players": [
                "Virat Kohli (Senior Batsman)",
                "Jasprit Bumrah (Lead Fast Bowler)", 
                "Ravindra Jadeja (All-rounder)",
                "Rohit Sharma (Captain/Opening Batsman)"
            ],
            "squad_strengths": [
                "Experienced spin bowling attack with local knowledge",
                "Deep batting lineup with proven big-match performers",
                "Home ground familiarity and crowd support",
                "Athletic fielding unit with strong catching record",
                "Versatile all-rounders providing tactical flexibility"
            ],
            "squad_weaknesses": [
                "Death bowling consistency in high-pressure situations",
                "Middle-order batting stability under pressure",
                "Tournament knockout stage performance anxiety"
            ],
            "opposition_analysis": {
                "opponent_strengths": [
                    "Aggressive power-hitting capability",
                    "Quality pace bowling attack",
                    "Big tournament experience and mental toughness"
                ],
                "opponent_weaknesses": [
                    "Spin bowling adaptation in subcontinent conditions",
                    "Middle overs scoring rate management",
                    "Unfamiliarity with local ground conditions"
                ]
            }
        },
        "performance_history": {
            "head_to_head_record": "India leads 12-8 in last 20 T20 internationals",
            "most_recent_encounter": {
                "date": "2024-01-15",
                "venue": "Melbourne Cricket Ground",
                "result": "Australia won by 7 runs",
                "key_performance_factors": [
                    "Australian pace bowling dominated powerplay",
                    "Indian middle-order collapse under pressure",
                    "Effective Australian death bowling strategy"
                ]
            },
            "tournament_progression": {
                "team_performance": "Group stage: 4 wins from 4 matches, Quarter-final: Defeated England by 8 wickets",
                "opponent_performance": "Group stage: 3 wins from 4 matches, Quarter-final: Defeated South Africa by 5 runs"
            }
        },
        "tactical_considerations": {
            "pitch_behavior_timeline": [
                "Overs 1-6: Pace-friendly with some movement",
                "Overs 7-12: Batting-friendly with minimal assistance",
                "Overs 13-20: Increasing spin assistance and variable bounce"
            ],
            "dew_factor_impact": "Expected from over 15 onwards, affecting ball grip and spin bowling",
            "crowd_noise_factor": "Significant home advantage with capacity crowd expected"
        }
    }

def save_strategic_analysis(analysis_results: Dict, filename: Optional[str] = None) -> str:
    """Save strategic analysis results with proper formatting"""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        teams = analysis_results["match_metadata"]["teams"].replace(" vs ", "_vs_")
        filename = f"cricket_strategic_analysis_{teams}_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        return filename
    except Exception as e:
        print(f"ERROR: Could not save analysis to {filename}: {e}")
        return ""

def print_strategic_summary(analysis_results: Dict) -> None:
    """Print formatted strategic analysis summary"""
    
    print("\n" + "=" * 80)
    print("CRICKET STRATEGIC ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Match metadata
    metadata = analysis_results["match_metadata"]
    strategic = analysis_results["strategic_analysis"]
    framework = analysis_results["strategic_framework"]
    
    print(f"Match: {metadata['teams']}")
    print(f"Tournament: {metadata['tournament']}")
    print(f"Venue: {metadata['venue']}")
    print(f"Analysis Date: {metadata['analysis_timestamp'][:19]}")
    
    print(f"\nSTRATEGIC ASSESSMENT:")
    print(f"   Success Feasibility: {strategic['success_feasibility_score']}")
    print(f"   Analysis Confidence: {strategic['confidence_rating']}")
    print(f"   Overall Approach: {framework['overall_strategic_approach']}")
    
    print(f"\nKEY STRATEGIC FACTORS:")
    for factor in strategic['key_strategic_factors']:
        print(f"   • {factor}")
    
    print(f"\nTACTICAL RECOMMENDATIONS:")
    for recommendation in strategic['tactical_recommendations']:
        print(f"   • {recommendation}")
    
    print(f"\nPRIORITY ACTION ITEMS:")
    for action in framework['priority_action_items']:
        print(f"   {action}")
    
    print(f"\nRISK ASSESSMENT:")
    for risk in framework['risk_assessment']:
        print(f"   • {risk}")
    
    print("\n" + "=" * 80)

# ============================================================================
# 3. MAIN EXECUTION FRAMEWORK
# ============================================================================

def main():
    """Main execution function for cricket strategic analysis"""
    
    print("CRICKET STRATEGIC FEASIBILITY ASSESSMENT SYSTEM")
    print("=" * 80)
    print("Professional strategic analysis for cricket match planning")
    
    # Generate comprehensive match data
    print("\nGenerating comprehensive match scenario data...")
    match_data = generate_comprehensive_match_data()
    
    # Save input data file
    input_filename = "cricket_match_scenario.json"
    try:
        with open(input_filename, 'w', encoding='utf-8') as f:
            json.dump(match_data, f, indent=2, ensure_ascii=False)
        print(f"Match scenario data saved: {input_filename}")
    except Exception as e:
        print(f"WARNING: Could not save input file: {e}")
    
    # Configure analysis platform
    llm_platform = "transformers"  # Options: "transformers" or "ollama"
    
    print(f"\nInitializing {llm_platform.upper()} strategic analysis platform...")
    
    try:
        # Initialize strategic analyzer
        analyzer = CricketStrategicAnalyzer(llm_type=llm_platform)
        
        # Execute strategic analysis
        print("\nExecuting comprehensive strategic analysis...")
        strategic_results = analyzer.analyze_match_strategy(match_data)
        
        # Save analysis results
        output_filename = save_strategic_analysis(strategic_results)
        if output_filename:
            print(f"Strategic analysis saved: {output_filename}")
        
        # Display comprehensive summary
        print_strategic_summary(strategic_results)
        
        # Display detailed strategic reasoning
        print("\nDETAILED STRATEGIC ANALYSIS:")
        print("-" * 80)
        llm_analysis = strategic_results["strategic_analysis"]["complete_llm_analysis"]
        if len(llm_analysis) > 800:
            print(llm_analysis[:800] + "...")
            print("\n[Analysis truncated - see output file for complete details]")
        else:
            print(llm_analysis)
        print("-" * 80)
        
        print(f"\nAnalysis workflow completed successfully.")
        print(f"Files generated:")
        print(f"   Input scenario: {input_filename}")
        if output_filename:
            print(f"   Strategic analysis: {output_filename}")
        
    except Exception as e:
        print(f"\nERROR: Strategic analysis failed: {e}")
        print("\nSystem requirements verification:")
        print("   For Transformers: pip install transformers torch")
        print("   For Ollama: Install from ollama.ai and run 'ollama serve'")
        print("   Ensure sufficient system memory for model operations")

def validate_analysis_quality(strategic_results: Dict) -> Dict:
    """Validate the quality and completeness of strategic analysis"""
    
    validation_report = {
        "analysis_completeness": "Complete",
        "data_quality_score": 0,
        "validation_issues": [],
        "recommendations_clarity": "High"
    }
    
    # Check analysis completeness
    required_sections = [
        "strategic_analysis", "strategic_framework", 
        "match_metadata", "environmental_context"
    ]
    
    missing_sections = [section for section in required_sections 
                       if section not in strategic_results]
    
    if missing_sections:
        validation_report["analysis_completeness"] = "Incomplete"
        validation_report["validation_issues"].extend(
            [f"Missing section: {section}" for section in missing_sections]
        )
    
    # Assess data quality
    strategic_analysis = strategic_results.get("strategic_analysis", {})
    factors_count = len(strategic_analysis.get("key_strategic_factors", []))
    recommendations_count = len(strategic_analysis.get("tactical_recommendations", []))
    
    quality_score = 0
    if factors_count >= 4:
        quality_score += 25
    if recommendations_count >= 4:
        quality_score += 25
    if len(strategic_analysis.get("complete_llm_analysis", "")) > 300:
        quality_score += 25
    if strategic_analysis.get("success_feasibility_score") in ["High", "Medium", "Low"]:
        quality_score += 25
    
    validation_report["data_quality_score"] = quality_score
    
    # Assess recommendations clarity
    if recommendations_count < 3:
        validation_report["recommendations_clarity"] = "Low"
        validation_report["validation_issues"].append("Insufficient tactical recommendations")
    elif recommendations_count < 5:
        validation_report["recommendations_clarity"] = "Medium"
    
    return validation_report

def generate_executive_summary(strategic_results: Dict) -> str:
    """Generate executive summary for strategic analysis"""
    
    metadata = strategic_results["match_metadata"]
    strategic = strategic_results["strategic_analysis"]
    framework = strategic_results["strategic_framework"]
    
    summary = f"""
EXECUTIVE SUMMARY - CRICKET STRATEGIC ANALYSIS

Match Overview:
{metadata['teams']} in {metadata['tournament']}
Venue: {metadata['venue']}
Analysis Confidence: {strategic['confidence_rating']}

Strategic Assessment:
Success Feasibility: {strategic['success_feasibility_score']}
Recommended Approach: {framework['overall_strategic_approach']}

Key Success Factors:
{chr(10).join(f'• {factor}' for factor in strategic['key_strategic_factors'][:3])}

Priority Actions:
{chr(10).join(f'{action}' for action in framework['priority_action_items'][:3])}

Risk Considerations:
{chr(10).join(f'• {risk}' for risk in framework['risk_assessment'][:2])}

This analysis provides a comprehensive strategic framework for match preparation
and tactical implementation based on environmental conditions, team capabilities,
and historical performance data.
"""
    
    return summary.strip()

def export_analysis_report(strategic_results: Dict, export_format: str = "json") -> str:
    """Export analysis in various formats"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    teams = strategic_results["match_metadata"]["teams"].replace(" vs ", "_vs_")
    
    if export_format == "json":
        filename = f"cricket_strategic_report_{teams}_{timestamp}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(strategic_results, f, indent=2, ensure_ascii=False)
            return filename
        except Exception as e:
            print(f"ERROR: Could not export JSON report: {e}")
            return ""
    
    elif export_format == "summary":
        filename = f"cricket_executive_summary_{teams}_{timestamp}.txt"
        try:
            summary = generate_executive_summary(strategic_results)
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(summary)
            return filename
        except Exception as e:
            print(f"ERROR: Could not export summary: {e}")
            return ""
    
    else:
        print(f"ERROR: Unsupported export format: {export_format}")
        return ""

if __name__ == "__main__":
    main()
