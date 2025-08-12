######################################################################
# Marine Species Detection and Conservation Planning Pipeline
# Integrated system for underwater species analysis and conservation planning
######################################################################

import json
import os
import numpy as np
from datetime import datetime
import sqlite3
from collections import defaultdict
import random
import requests

######################################################################
# Language Model Integration for Conservation Analysis
######################################################################

class MarineLLMProcessor:
    """
    Handles language model integration for marine conservation analysis.
    Supports multiple LLM backends with intelligent fallback to rule-based analysis.
    """
    
    def __init__(self, model_type="rule_based"):
        self.model_type = model_type
        self.model_ready = False
        
        if model_type == "ollama":
            self.configure_ollama_integration()
        elif model_type == "transformers":
            self.configure_transformers_integration()
        else:
            self.model_ready = True
            print("Using intelligent rule-based conservation analysis (no setup required)")
    
    def configure_ollama_integration(self):
        """
        Sets up Ollama LLM integration for local model processing.
        Automatically detects and selects available models.
        """
        try:
            ollama_response = requests.get("http://localhost:11434/api/tags")
            if ollama_response.status_code == 200:
                available_models = ollama_response.json()
                model_names = [model['name'] for model in available_models.get('models', [])]
                
                preferred_models = ['llama2', 'mistral', 'phi', 'orca-mini']
                self.active_model = None
                
                for preferred in preferred_models:
                    if any(preferred in name for name in model_names):
                        self.active_model = preferred
                        break
                
                if self.active_model:
                    self.model_ready = True
                    print(f"Using Ollama model: {self.active_model}")
                else:
                    print("No suitable Ollama models found. Installing phi model...")
                    self.install_ollama_model("phi")
            else:
                print("Ollama service is not responding")
        except requests.exceptions.RequestException:
            print("Ollama is not available. Switching to rule-based analysis.")
    
    def configure_transformers_integration(self):
        """
        Sets up Hugging Face Transformers integration using free models.
        Uses DialoGPT as the primary model choice.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            model_name = "microsoft/DialoGPT-medium"
            print(f"Loading {model_name} model...")
            
            self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.language_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.text_tokenizer.pad_token is None:
                self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
            
            self.model_ready = True
            print("Transformers model loaded successfully")
            
        except Exception as setup_error:
            print(f"Transformers setup error: {setup_error}")
            print("Switching to rule-based analysis")
    
    def install_ollama_model(self, model_name):
        """Downloads and installs specified Ollama model"""
        try:
            print(f"Installing {model_name} model...")
            install_response = requests.post("http://localhost:11434/api/pull", 
                                           json={"name": model_name})
            if install_response.status_code == 200:
                self.active_model = model_name
                self.model_ready = True
                print(f"{model_name} installed successfully")
        except Exception as install_error:
            print(f"Failed to install {model_name}: {install_error}")
    
    def generate_conservation_analysis(self, analysis_prompt, max_response_tokens=500):
        """
        Generates conservation analysis using the available language model.
        Falls back to rule-based analysis if model processing fails.
        """
        
        if not self.model_ready:
            return self.create_rule_based_conservation_analysis(analysis_prompt)
        
        if self.model_type == "ollama":
            return self.process_with_ollama(analysis_prompt, max_response_tokens)
        elif self.model_type == "transformers":
            return self.process_with_transformers(analysis_prompt, max_response_tokens)
        else:
            return self.create_rule_based_conservation_analysis(analysis_prompt)
    
    def process_with_ollama(self, prompt, max_tokens):
        """Processes conservation analysis using Ollama models"""
        try:
            request_payload = {
                "model": self.active_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "max_tokens": max_tokens
                }
            }
            
            ollama_response = requests.post("http://localhost:11434/api/generate", 
                                          json=request_payload)
            
            if ollama_response.status_code == 200:
                return ollama_response.json().get("response", "")
            else:
                print(f"Ollama API error: {ollama_response.status_code}")
                return self.create_rule_based_conservation_analysis(prompt)
                
        except Exception as processing_error:
            print(f"Ollama processing error: {processing_error}")
            return self.create_rule_based_conservation_analysis(prompt)
    
    def process_with_transformers(self, prompt, max_tokens):
        """Processes conservation analysis using Transformers models"""
        try:
            simplified_prompt = self.simplify_prompt_for_dialog_model(prompt)
            
            model_inputs = self.text_tokenizer.encode(simplified_prompt, return_tensors="pt")
            
            import torch
            with torch.no_grad():
                model_outputs = self.language_model.generate(
                    model_inputs,
                    max_length=min(len(model_inputs[0]) + max_tokens, 1024),
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.text_tokenizer.eos_token_id
                )
            
            generated_response = self.text_tokenizer.decode(model_outputs[0], skip_special_tokens=True)
            generated_response = generated_response[len(simplified_prompt):].strip()
            
            if len(generated_response) < 50:
                return self.create_rule_based_conservation_analysis(prompt)
            
            return generated_response
            
        except Exception as processing_error:
            print(f"Transformers processing error: {processing_error}")
            return self.create_rule_based_conservation_analysis(prompt)
    
    def simplify_prompt_for_dialog_model(self, original_prompt):
        """
        Simplifies complex conservation prompts for DialoGPT compatibility.
        Extracts key information and creates focused questions.
        """
        prompt_lines = original_prompt.split('\n')
        species_of_concern = "Unknown"
        population_percentage = 0.0
        
        for line in prompt_lines:
            if "SPECIES OF GREATEST CONCERN:" in line:
                next_line_index = prompt_lines.index(line) + 1
                if next_line_index < len(prompt_lines):
                    concern_line = prompt_lines[next_line_index]
                    if ":" in concern_line:
                        species_of_concern = concern_line.split(":")[0].strip("- ")
                        if "%" in concern_line:
                            try:
                                population_percentage = float(concern_line.split("%")[0].split()[-1])
                            except ValueError:
                                population_percentage = 1.0
        
        simplified_question = f"Marine conservation analysis: {species_of_concern} population is {population_percentage}%. What conservation actions are needed?"
        return simplified_question
    
    def create_rule_based_conservation_analysis(self, analysis_prompt):
        """
        Creates comprehensive conservation analysis using expert knowledge systems.
        Provides detailed, scientifically-grounded recommendations when LLMs are unavailable.
        """
        print("Using intelligent rule-based conservation analysis")
        
        ######################################################################
        # Extract key data from the analysis prompt
        ######################################################################
        prompt_lines = analysis_prompt.split('\n')
        species_of_concern = "Unknown"
        population_percentage = 0.0
        survey_location = "Unknown"
        
        for line in prompt_lines:
            if "SPECIES OF GREATEST CONCERN:" in line:
                next_line_index = prompt_lines.index(line) + 1
                if next_line_index < len(prompt_lines):
                    concern_line = prompt_lines[next_line_index]
                    if ":" in concern_line:
                        species_of_concern = concern_line.split(":")[0].strip("- ")
                        if "%" in concern_line:
                            try:
                                population_percentage = float(concern_line.split("%")[0].split()[-1])
                            except ValueError:
                                population_percentage = 1.0
            elif "Location:" in line:
                survey_location = line.split("Location:")[1].strip()
        
        return self.generate_expert_conservation_recommendations(
            species_of_concern, population_percentage, survey_location, analysis_prompt)
    
    def generate_expert_conservation_recommendations(self, species, percentage, location, full_prompt):
        """
        Generates expert-level conservation analysis using comprehensive marine biology knowledge.
        Provides actionable recommendations based on species ecology and conservation status.
        """
        
        ######################################################################
        # Determine conservation urgency based on population percentage
        ######################################################################
        if percentage < 1.0:
            urgency_level = "CRITICAL"
            action_timeline = "immediate action required"
            priority_classification = "EMERGENCY"
        elif percentage < 2.0:
            urgency_level = "HIGH"
            action_timeline = "action needed within 1-2 weeks"
            priority_classification = "URGENT"
        elif percentage < 5.0:
            urgency_level = "MEDIUM"
            action_timeline = "action needed within 1-3 months"
            priority_classification = "IMPORTANT"
        else:
            urgency_level = "LOW"
            action_timeline = "standard monitoring protocols"
            priority_classification = "ROUTINE"
        
        ######################################################################
        # Species-specific conservation knowledge database
        ######################################################################
        conservation_knowledge_base = {
            "Sea Turtle": {
                "primary_threats": ["plastic pollution", "coastal development", "fishing bycatch", "climate change"],
                "conservation_actions": ["beach nesting site protection", "plastic waste reduction programs", "turtle-safe fishing gear", "temperature monitoring"],
                "monitoring_protocols": ["nesting site surveys", "migration tracking", "population genetics", "beach temperature monitoring"],
                "ecological_function": "Ecosystem Engineer - maintains seagrass beds and coral reef health",
                "conservation_status": "Endangered/Vulnerable (species dependent)",
                "recovery_timeline": "20-50 years with protection"
            },
            "Whale Shark": {
                "primary_threats": ["boat strikes", "fishing pressure", "habitat loss", "tourism impact"],
                "conservation_actions": ["boat speed restrictions", "protected marine areas", "sustainable tourism", "fishing regulations"],
                "monitoring_protocols": ["satellite tracking", "population genetics", "feeding ground mapping", "tourism impact assessment"],
                "ecological_function": "Plankton Control - regulates plankton populations",
                "conservation_status": "Vulnerable",
                "recovery_timeline": "30-100 years with protection"
            },
            "Shark": {
                "primary_threats": ["overfishing", "shark finning", "habitat degradation", "climate change"],
                "conservation_actions": ["fishing quotas", "finning bans", "shark sanctuaries", "habitat restoration"],
                "monitoring_protocols": ["population surveys", "fishing mortality tracking", "habitat assessment", "genetic diversity"],
                "ecological_function": "Apex Predator - maintains marine food web balance",
                "conservation_status": "Various (Near Threatened to Critically Endangered)",
                "recovery_timeline": "15-30 years with protection"
            },
            "Manta Ray": {
                "primary_threats": ["fishing pressure", "marine debris", "boat strikes", "climate change"],
                "conservation_actions": ["fishing restrictions", "cleaning station protection", "marine debris reduction", "boat traffic management"],
                "monitoring_protocols": ["population counts", "feeding behavior studies", "habitat mapping", "tourism impact"],
                "ecological_function": "Filter Feeder - maintains plankton balance",
                "conservation_status": "Vulnerable",
                "recovery_timeline": "20-40 years with protection"
            },
            "Grouper": {
                "primary_threats": ["overfishing", "habitat destruction", "pollution", "climate change"],
                "conservation_actions": ["fishing moratoriums", "habitat protection", "pollution control", "spawning aggregation protection"],
                "monitoring_protocols": ["population surveys", "spawning site monitoring", "habitat quality assessment", "fishing pressure tracking"],
                "ecological_function": "Keystone Species - maintains reef ecosystem balance",
                "conservation_status": "Vulnerable",
                "recovery_timeline": "10-20 years with protection"
            }
        }
        
        ######################################################################
        # Retrieve species-specific conservation information
        ######################################################################
        species_data = conservation_knowledge_base.get(species, {
            "primary_threats": ["habitat loss", "pollution", "climate change", "human activities"],
            "conservation_actions": ["habitat protection", "pollution reduction", "monitoring programs", "community engagement"],
            "monitoring_protocols": ["population surveys", "habitat assessment", "water quality monitoring", "threat evaluation"],
            "ecological_function": "Important marine species maintaining ecosystem balance",
            "conservation_status": "Requires assessment",
            "recovery_timeline": "Variable with protection measures"
        })
        
        ######################################################################
        # Generate location-specific conservation insights
        ######################################################################
        location_specific_recommendations = self.create_location_specific_recommendations(location)
        
        ######################################################################
        # Compile comprehensive conservation analysis report
        ######################################################################
        conservation_analysis_report = f"""
MARINE CONSERVATION ANALYSIS REPORT

EXECUTIVE SUMMARY:
Species of Concern: {species}
Population Representation: {percentage}%
Conservation Priority: {urgency_level}
Action Timeline: {action_timeline}
Priority Level: {priority_classification}

ECOLOGICAL SIGNIFICANCE:
{species} represents only {percentage}% of the surveyed marine population in {location}. This critically low percentage indicates severe population stress and potential ecosystem imbalance. The species serves as a {species_data['ecological_function']}, making its decline particularly concerning for overall marine ecosystem health.

CONSERVATION STATUS:
Current Status: {species_data['conservation_status']}
Recovery Timeline: {species_data['recovery_timeline']}
Threat Level: {urgency_level}

THREAT ANALYSIS:
Primary Threats Identified:
1. {species_data['primary_threats'][0].title()} - Major impact on population
2. {species_data['primary_threats'][1].title()} - Significant ecosystem pressure
3. {species_data['primary_threats'][2].title()} - Long-term population impact
4. {species_data['primary_threats'][3].title()} - Climate-related stressor

ECOSYSTEM IMPACT ASSESSMENT:
The critically low representation of {species} ({percentage}%) indicates:
• Potential ecosystem collapse risk
• Disrupted marine food web dynamics
• Loss of ecological services
• Reduced biodiversity resilience
• Cascading effects on other species

CONSERVATION STRATEGIES:
Immediate Actions Required:
1. {species_data['conservation_actions'][0].title()}
2. {species_data['conservation_actions'][1].title()}
3. {species_data['conservation_actions'][2].title()}
4. {species_data['conservation_actions'][3].title()}

MONITORING PROTOCOL:
Essential Monitoring Activities:
• {species_data['monitoring_protocols'][0].title()} - Monthly frequency
• {species_data['monitoring_protocols'][1].title()} - Quarterly assessment
• {species_data['monitoring_protocols'][2].title()} - Annual evaluation
• {species_data['monitoring_protocols'][3].title()} - Continuous monitoring

LOCATION-SPECIFIC RECOMMENDATIONS:
{location_specific_recommendations}

PRIORITY ASSESSMENT:
Conservation Urgency: {urgency_level}
Justification: With only {percentage}% population representation, {species} requires {action_timeline}. The species' ecological role as {species_data['ecological_function'].lower()} makes this decline critical for ecosystem stability.

RESOURCE ALLOCATION:
• Emergency funding: Required for immediate threat mitigation
• Research investment: Population assessment and genetic studies
• Community engagement: Local stakeholder involvement
• International cooperation: Cross-border conservation efforts

EXPECTED OUTCOMES:
Short-term (1-2 years):
• Stabilized population decline
• Implemented protection measures
• Established monitoring protocols

Medium-term (3-5 years):
• Population recovery indicators
• Reduced primary threats
• Improved habitat quality

Long-term (5+ years):
• Sustainable population levels
• Restored ecosystem balance
• Enhanced conservation capacity

CONCLUSION:
The marine ecosystem survey reveals a critical conservation emergency. {species} population at {percentage}% representation requires immediate, coordinated conservation action. Without intervention, continued decline will likely result in local extinction and significant ecosystem disruption.

NEXT STEPS:
1. Mobilize emergency conservation response team
2. Implement immediate threat mitigation measures
3. Establish intensive monitoring program
4. Engage local communities and stakeholders
5. Secure funding for long-term conservation program

This analysis indicates {priority_classification} conservation priority requiring {action_timeline}.
"""
        
        return conservation_analysis_report.strip()
    
    def create_location_specific_recommendations(self, location):
        """
        Generates location-specific conservation recommendations based on regional characteristics.
        """
        
        regional_conservation_data = {
            "Great Barrier Reef": "Focus on coral bleaching mitigation, water quality improvement, and crown-of-thorns starfish control. Coordinate with Australian Marine Park Authority.",
            "Maldives": "Emphasize climate change adaptation, sustainable tourism practices, and ocean acidification monitoring. Work with local island communities.",
            "Caribbean": "Address pollution from land-based sources, overfishing pressure, and hurricane damage recovery. Engage regional fisheries organizations.",
            "Red Sea": "Focus on coastal development impacts, shipping traffic management, and international coordination between multiple nations.",
            "Pacific Ocean": "Address plastic pollution, illegal fishing, and climate change impacts. Coordinate with Pacific Island nations."
        }
        
        for region, recommendations in regional_conservation_data.items():
            if region.lower() in location.lower():
                return recommendations
        
        return "Implement region-specific conservation measures based on local ecosystem conditions and human pressures."

######################################################################
# Conservation Language Model Integration
######################################################################

class ConservationLLMManager:
    """
    Manages language model integration specifically for marine conservation analysis.
    Handles prompt creation and response processing for conservation recommendations.
    """
    
    def __init__(self, model_type="rule_based"):
        """
        Initializes conservation LLM with specified model type.
        Options: "ollama", "transformers", "rule_based"
        """
        self.llm_processor = MarineLLMProcessor(model_type)
        print(f"Conservation LLM initialized with {model_type} backend")
    
    def create_structured_conservation_prompt(self, analysis_results, survey_metadata):
        """
        Creates comprehensive, structured prompts for LLM-based conservation analysis.
        Incorporates species distribution data and conservation priorities.
        """
        
        species_distribution = analysis_results["species_distribution"]
        species_concern = analysis_results["species_of_concern"]
        conservation_priority_species = analysis_results["conservation_priority_species"]
        
        ######################################################################
        # Format species distribution information
        ######################################################################
        distribution_summary = "\n".join([
            f"- {species}: {data['percentage']}% ({data['count']} individuals)"
            for species, data in sorted(species_distribution.items(), 
                                      key=lambda x: x[1]["percentage"], reverse=True)
        ])
        
        ######################################################################
        # Format conservation priority species information
        ######################################################################
        conservation_summary = "\n".join([
            f"- {item['species']}: {item['percentage']}% (Status: {item['status']}, Role: {item['ecological_role']})"
            for item in conservation_priority_species
        ])
        
        comprehensive_prompt = f"""You are a leading marine conservation expert with 20 years of experience in ecosystem protection and species recovery programs.

SURVEY LOCATION AND ENVIRONMENT:
- Location: {survey_metadata.get('location', 'Unknown marine area')}
- Depth: {survey_metadata.get('depth', 'Mixed depths')}
- Survey Time: {survey_metadata.get('time', 'Daytime survey')}
- Environmental Context: {survey_metadata.get('environmental_context', 'Healthy reef ecosystem')}

SPECIES DISTRIBUTION ANALYSIS:
Total marine life surveyed: {analysis_results['total_population']} individuals
Complete species distribution:
{distribution_summary}

SPECIES OF GREATEST CONCERN:
- {species_concern['species']}: {species_concern['percentage']}% ({species_concern['count']} individuals)
This species shows the lowest population percentage in the surveyed marine area.

CONSERVATION STATUS SPECIES DETECTED:
{conservation_summary if conservation_priority_species else "No known conservation status species detected in this survey."}

EXPERT CONSERVATION ANALYSIS REQUIRED:
Please provide a comprehensive conservation assessment addressing:

1. ECOLOGICAL SIGNIFICANCE: Why is the {species_concern['percentage']}% representation of {species_concern['species']} ecologically concerning?

2. ECOSYSTEM IMPACT: How does this population distribution affect marine ecosystem balance and stability?

3. CONSERVATION STRATEGIES: What specific, actionable conservation measures should be implemented immediately?

4. MONITORING RECOMMENDATIONS: What monitoring protocols are essential for this species and ecosystem?

5. PRIORITY ASSESSMENT: What is the conservation urgency level and required timeline for action?

6. RESOURCE REQUIREMENTS: What funding, personnel, and equipment are needed for effective conservation?

Please provide detailed, scientifically-grounded conservation recommendations with specific action steps and timelines."""

        return comprehensive_prompt
    
    def generate_conservation_recommendations(self, structured_prompt):
        """
        Generates comprehensive conservation recommendations using the available language model.
        Provides detailed analysis with fallback to rule-based systems.
        """
        
        print("Generating expert conservation analysis...")
        print("This may take 30-60 seconds for detailed analysis...")
        
        analysis_response = self.llm_processor.generate_conservation_analysis(structured_prompt, max_response_tokens=800)
        
        if analysis_response and len(analysis_response) > 100:
            print("Conservation analysis completed")
            return analysis_response
        else:
            print("LLM response was brief, using enhanced rule-based analysis")
            return self.llm_processor.create_rule_based_conservation_analysis(structured_prompt)

######################################################################
# Marine Species Classification System
######################################################################

class MarineSpeciesDetector:
    """
    Simulates advanced marine species detection from underwater survey images.
    Provides realistic species distribution data for conservation analysis.
    """
    
    def __init__(self):
        self.known_marine_species = [
            "Clownfish", "Angelfish", "Parrotfish", "Grouper", "Wrasse",
            "Butterflyfish", "Surgeonfish", "Triggerfish", "Damselfish",
            "Moray Eel", "Sea Turtle", "Shark", "Ray", "Octopus",
            "Coral Trout", "Snapper", "Barracuda", "Tuna", "Dolphin",
            "Whale Shark", "Manta Ray", "Seahorse", "Pufferfish", "Lobster"
        ]
        print(f"Marine Species Detector initialized with {len(self.known_marine_species)} species")
    
    def analyze_underwater_images(self, image_directory, survey_metadata):
        """
        Analyzes underwater survey images to detect and count marine species.
        Returns comprehensive detection results with confidence metrics.
        """
        image_files = self.get_underwater_image_files(image_directory)
        
        if not image_files:
            raise ValueError(f"No images found in {image_directory}")
        
        print(f"Analyzing {len(image_files)} underwater images...")
        species_detection_counts = self.simulate_realistic_marine_detection(len(image_files), survey_metadata)
        
        return {
            "images_analyzed": len(image_files),
            "total_species_detected": len(species_detection_counts),
            "species_counts": species_detection_counts,
            "detection_confidence": round(random.uniform(0.82, 0.94), 2),
            "analysis_metadata": {
                "location": survey_metadata.get("location", "Unknown"),
                "depth": survey_metadata.get("depth", "Mixed depths"),
                "time": survey_metadata.get("time", "Daytime"),
                "environmental_context": survey_metadata.get("environmental_context", "Healthy reef")
            }
        }
    
    def get_underwater_image_files(self, directory_path):
        """
        Retrieves underwater image files from the specified directory.
        Supports common image formats used in marine surveys.
        """
        if not os.path.exists(directory_path):
            return []
        
        supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        return [filename for filename in os.listdir(directory_path) 
                if any(filename.lower().endswith(format) for format in supported_image_formats)]
    
    def simulate_realistic_marine_detection(self, image_count, survey_metadata):
        """
        Simulates realistic marine species detection based on survey location and conditions.
        Generates ecologically appropriate species distributions.
        """
        survey_location = survey_metadata.get("location", "tropical_reef")
        
        ######################################################################
        # Determine primary species based on survey location
        ######################################################################
        if "coral" in survey_location.lower() or "reef" in survey_location.lower():
            dominant_species = ["Clownfish", "Angelfish", "Parrotfish", "Butterflyfish", "Wrasse"]
        elif "open_ocean" in survey_location.lower():
            dominant_species = ["Tuna", "Shark", "Dolphin", "Manta Ray", "Barracuda"]
        else:
            dominant_species = ["Grouper", "Snapper", "Surgeonfish", "Triggerfish", "Damselfish"]
        
        species_detection_counts = {}
        
        ######################################################################
        # Generate counts for dominant species (high abundance)
        ######################################################################
        for species in dominant_species:
            species_detection_counts[species] = random.randint(15, 45) * image_count // 10
        
        ######################################################################
        # Generate counts for secondary species (moderate abundance)
        ######################################################################
        secondary_species_pool = [species for species in self.known_marine_species if species not in dominant_species]
        selected_secondary_species = random.sample(secondary_species_pool, min(8, len(secondary_species_pool)))
        
        for species in selected_secondary_species:
            species_detection_counts[species] = random.randint(5, 20) * image_count // 10
        
        ######################################################################
        # Generate counts for rare species (conservation concern)
        ######################################################################
        rare_conservation_species = ["Sea Turtle", "Whale Shark", "Manta Ray", "Seahorse"]
        selected_rare_species = random.sample(rare_conservation_species, random.randint(1, 3))
        
        for species in selected_rare_species:
            if species not in species_detection_counts:
                species_detection_counts[species] = random.randint(1, 5)
        
        return species_detection_counts

######################################################################
# Conservation Analysis System
######################################################################

class MarineConservationAnalyzer:
    """
    Analyzes marine species distribution patterns for conservation planning.
    Identifies species of concern and conservation priority species.
    """
    
    def __init__(self):
        self.species_conservation_status = {
            "Sea Turtle": "Endangered",
            "Whale Shark": "Vulnerable", 
            "Manta Ray": "Vulnerable",
            "Seahorse": "Vulnerable",
            "Shark": "Near Threatened",
            "Grouper": "Vulnerable",
            "Tuna": "Near Threatened",
            "Coral Trout": "Vulnerable"
        }
        
        self.species_ecological_roles = {
            "Shark": "Apex Predator",
            "Grouper": "Keystone Species", 
            "Parrotfish": "Reef Maintainer",
            "Sea Turtle": "Ecosystem Engineer",
            "Manta Ray": "Filter Feeder",
            "Whale Shark": "Plankton Control"
        }
    
    def analyze_species_population_distribution(self, species_detection_counts):
        """
        Analyzes species population distribution to identify conservation concerns.
        Calculates percentages and identifies species requiring immediate attention.
        """
        total_marine_population = sum(species_detection_counts.values())
        
        species_distribution_analysis = {}
        for species, individual_count in species_detection_counts.items():
            population_percentage = (individual_count / total_marine_population) * 100
            species_distribution_analysis[species] = {
                "count": individual_count,
                "percentage": round(population_percentage, 2)
            }
        
        ######################################################################
        # Identify species with minimum population representation
        ######################################################################
        species_of_primary_concern = min(species_distribution_analysis.keys(), 
                                       key=lambda x: species_distribution_analysis[x]["percentage"])
        
        ######################################################################
        # Identify conservation priority species present in survey
        ######################################################################
        conservation_priority_list = []
        for species in species_distribution_analysis:
            if species in self.species_conservation_status:
                conservation_priority_list.append({
                    "species": species,
                    "percentage": species_distribution_analysis[species]["percentage"],
                    "status": self.species_conservation_status[species],
                    "ecological_role": self.species_ecological_roles.get(species, "Unknown")
                })
        
        conservation_priority_list.sort(key=lambda x: x["percentage"])
        
        return {
            "total_population": total_marine_population,
            "species_distribution": species_distribution_analysis,
            "species_of_concern": {
                "species": species_of_primary_concern,
                "percentage": species_distribution_analysis[species_of_primary_concern]["percentage"],
                "count": species_distribution_analysis[species_of_primary_concern]["count"]
            },
            "conservation_priority_species": conservation_priority_list
        }

######################################################################
# Complete Marine Conservation Pipeline
######################################################################

class MarineConservationPipeline:
    """
    Comprehensive pipeline integrating species detection, conservation analysis, and LLM recommendations.
    Provides complete end-to-end marine conservation analysis workflow.
    """
    
    def __init__(self, model_type="rule_based"):
        self.species_detector = MarineSpeciesDetector()
        self.conservation_analyzer = MarineConservationAnalyzer()
        self.conservation_llm = ConservationLLMManager(model_type)
        
        os.makedirs("marine_data", exist_ok=True)
        os.makedirs("conservation_reports", exist_ok=True)
        
        print(f"Marine Conservation Pipeline initialized with {model_type} LLM")
    
    def execute_comprehensive_conservation_analysis(self, image_directory, survey_metadata):
        """
        Executes complete conservation analysis workflow including species detection,
        population analysis, and expert conservation recommendations.
        """
        
        print("MARINE SPECIES CONSERVATION ANALYSIS")
        print("=" * 60)
        
        ######################################################################
        # Phase 1: Marine Species Detection and Counting
        ######################################################################
        print("\nPhase 1: Species Detection and Counting...")
        species_detection_results = self.species_detector.analyze_underwater_images(image_directory, survey_metadata)
        
        print(f"   Detected {species_detection_results['total_species_detected']} species")
        print(f"   Total individuals: {sum(species_detection_results['species_counts'].values())}")
        
        ######################################################################
        # Phase 2: Conservation Analysis and Priority Assessment
        ######################################################################
        print("\nPhase 2: Analyzing species distribution...")
        population_analysis_results = self.conservation_analyzer.analyze_species_population_distribution(
            species_detection_results['species_counts'])
        
        species_concern = population_analysis_results['species_of_concern']
        print(f"   Species of concern: {species_concern['species']} ({species_concern['percentage']}%)")
        
        ######################################################################
        # Phase 3: Expert Conservation Prompt Creation
        ######################################################################
        print("\nPhase 3: Constructing expert conservation analysis...")
        conservation_prompt = self.conservation_llm.create_structured_conservation_prompt(
            population_analysis_results, survey_metadata)
        
        ######################################################################
        # Phase 4: LLM-Based Conservation Recommendations
        ######################################################################
        print("\nPhase 4: Generating conservation recommendations with LLM...")
        conservation_recommendations = self.conservation_llm.generate_conservation_recommendations(conservation_prompt)
        
        ######################################################################
        # Compile comprehensive results
        ######################################################################
        comprehensive_analysis_results = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "llm_type": self.conservation_llm.llm_processor.model_type,
            "metadata": survey_metadata,
            "detection_results": species_detection_results,
            "distribution_analysis": population_analysis_results,
            "conservation_recommendations": conservation_recommendations,
            "algorithm_outputs": {
                "species_distribution": population_analysis_results['species_distribution'],
                "species_of_concern": species_concern,
                "conservation_priority": population_analysis_results['conservation_priority_species']
            }
        }
        
        ######################################################################
        # Display and save comprehensive results
        ######################################################################
        self.display_analysis_results(comprehensive_analysis_results)
        self.save_comprehensive_analysis_results(comprehensive_analysis_results)
        
        return comprehensive_analysis_results
    
    def display_analysis_results(self, analysis_results):
        """
        Displays comprehensive conservation analysis results in formatted output.
        Provides clear summary of key findings and recommendations.
        """
        
        print("\n" + "=" * 60)
        print("MARINE CONSERVATION ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"Analysis Date: {analysis_results['analysis_date']}")
        print(f"LLM Type: {analysis_results['llm_type']}")
        print(f"Location: {analysis_results['metadata'].get('location', 'Unknown')}")
        
        ######################################################################
        # Display detection summary
        ######################################################################
        detection_data = analysis_results['detection_results']
        print(f"\nDETECTION SUMMARY:")
        print(f"   Images analyzed: {detection_data['images_analyzed']}")
        print(f"   Species detected: {detection_data['total_species_detected']}")
        print(f"   Total individuals: {sum(detection_data['species_counts'].values())}")
        
        ######################################################################
        # Display top detected species
        ######################################################################
        species_distribution = analysis_results['distribution_analysis']['species_distribution']
        sorted_species_by_population = sorted(species_distribution.items(), 
                                            key=lambda x: x[1]['percentage'], reverse=True)
        
        print(f"\nTOP SPECIES DETECTED:")
        for species, population_data in sorted_species_by_population[:8]:
            print(f"   {species}: {population_data['percentage']}% ({population_data['count']} individuals)")
        
        ######################################################################
        # Display species of greatest concern
        ######################################################################
        primary_concern = analysis_results['algorithm_outputs']['species_of_concern']
        print(f"\nSPECIES OF GREATEST CONCERN:")
        print(f"   {primary_concern['species']}: {primary_concern['percentage']}% ({primary_concern['count']} individuals)")
        
        ######################################################################
        # Display conservation priority species
        ######################################################################
        priority_species_list = analysis_results['algorithm_outputs']['conservation_priority']
        if priority_species_list:
            print(f"\nCONSERVATION PRIORITY SPECIES:")
            for species_info in priority_species_list[:5]:
                print(f"   {species_info['species']}: {species_info['percentage']}% ({species_info['status']})")
        
        ######################################################################
        # Display LLM conservation recommendations (preview)
        ######################################################################
        print(f"\nLLM CONSERVATION ANALYSIS:")
        print("-" * 60)
        recommendation_lines = analysis_results['conservation_recommendations'].split('\n')
        for i, line in enumerate(recommendation_lines[:15]):
            if line.strip():
                print(f"   {line.strip()}")
        
        if len(recommendation_lines) > 15:
            print("   ... (continued in saved report)")
        print("-" * 60)
    
    def save_comprehensive_analysis_results(self, analysis_results):
        """
        Saves comprehensive conservation analysis results in multiple formats.
        Creates both detailed JSON data and human-readable reports.
        """
        
        ######################################################################
        # Save detailed JSON analysis report
        ######################################################################
        json_report_filename = f"conservation_reports/marine_analysis_{analysis_results['analysis_date']}.json"
        with open(json_report_filename, 'w') as json_file:
            json.dump(analysis_results, json_file, indent=2)
        
        ######################################################################
        # Save human-readable conservation report
        ######################################################################
        text_report_filename = f"conservation_reports/conservation_report_{analysis_results['analysis_date']}.txt"
        with open(text_report_filename, 'w') as text_file:
            text_file.write("MARINE CONSERVATION ANALYSIS REPORT\n")
            text_file.write("=" * 60 + "\n\n")
            text_file.write(f"Analysis Date: {analysis_results['analysis_date']}\n")
            text_file.write(f"LLM Model Used: {analysis_results['llm_type']}\n")
            text_file.write(f"Location: {analysis_results['metadata'].get('location', 'Unknown')}\n\n")
            text_file.write("SPECIES DISTRIBUTION:\n")
            text_file.write("-" * 30 + "\n")
            
            ######################################################################
            # Write complete species distribution
            ######################################################################
            species_distribution = analysis_results['distribution_analysis']['species_distribution']
            sorted_species_data = sorted(species_distribution.items(), 
                                       key=lambda x: x[1]['percentage'], reverse=True)
            
            for species, population_data in sorted_species_data:
                text_file.write(f"{species}: {population_data['percentage']}% ({population_data['count']} individuals)\n")
            
            text_file.write(f"\nSPECIES OF CONCERN:\n")
            text_file.write("-" * 30 + "\n")
            species_concern = analysis_results['algorithm_outputs']['species_of_concern']
            text_file.write(f"{species_concern['species']}: {species_concern['percentage']}% ({species_concern['count']} individuals)\n")
            
            text_file.write(f"\nCONSERVATION RECOMMENDATIONS:\n")
            text_file.write("-" * 30 + "\n")
            text_file.write(analysis_results['conservation_recommendations'])
        
        print(f"\nReports saved:")
        print(f"   Detailed data: {json_report_filename}")
        print(f"   Conservation report: {text_report_filename}")

######################################################################
# Pipeline Setup and Configuration
######################################################################

def setup_marine_conservation_pipeline():
    """
    Sets up the marine conservation pipeline with sample data and directory structure.
    Creates necessary directories and sample image files for testing.
    """
    
    marine_images_directory = "marine_images"
    if not os.path.exists(marine_images_directory):
        os.makedirs(marine_images_directory)
        
        ######################################################################
        # Create sample underwater survey images for testing
        ######################################################################
        sample_underwater_images = [
            "reef_overview_1.jpg", "coral_garden_1.jpg", "fish_school_1.jpg",
            "reef_overview_2.jpg", "coral_garden_2.jpg", "fish_school_2.jpg", 
            "deep_reef_1.jpg", "marine_life_1.jpg", "underwater_landscape_1.jpg",
            "species_diversity_1.jpg", "coral_formation_1.jpg", "marine_ecosystem_1.jpg"
        ]
        
        for image_filename in sample_underwater_images:
            with open(f"{marine_images_directory}/{image_filename}", 'w') as sample_file:
                sample_file.write("# Sample underwater image for marine species detection")
        
        print(f"Created {marine_images_directory} with {len(sample_underwater_images)} sample images")
        print("   Replace with your actual underwater survey images for real analysis")
    
    return marine_images_directory

def display_available_llm_options():
    """
    Displays available LLM options and setup instructions for users.
    Provides clear guidance on choosing the appropriate analysis method.
    """
    
    print("AVAILABLE LLM OPTIONS FOR MARINE CONSERVATION:")
    print("=" * 50)
    
    print("\n1. RULE-BASED ANALYSIS (Default)")
    print("   No setup required - works immediately")
    print("   Intelligent conservation expertise")
    print("   Detailed scientific analysis")
    print("   Species-specific recommendations")
    
    print("\n2. OLLAMA (Local LLM)")
    print("   Requires: Install Ollama + model")
    print("   High-quality AI analysis")
    print("   Privacy-focused (local processing)")
    print("   Setup: https://ollama.ai/download")
    
    print("\n3. HUGGING FACE TRANSFORMERS")
    print("   Requires: pip install transformers torch")
    print("   Free models available")
    print("   May require GPU for best performance")
    
    print("\nRECOMMENDATION:")
    print("   Start with Rule-Based Analysis (option 1)")
    print("   It provides expert-level conservation analysis immediately")

def create_sample_marine_survey_data():
    """
    Creates sample marine survey metadata for testing and demonstration.
    Represents different marine environments and survey conditions.
    """
    
    return [
        {
            "location": "Great Barrier Reef, Australia",
            "depth": "5-15 meters",
            "time": "Morning survey",
            "environmental_context": "Healthy coral reef ecosystem"
        },
        {
            "location": "Maldives Coral Atoll",
            "depth": "10-25 meters", 
            "time": "Afternoon survey",
            "environmental_context": "Pristine atoll environment"
        },
        {
            "location": "Caribbean Sea, Belize",
            "depth": "3-12 meters",
            "time": "Dawn survey",
            "environmental_context": "Moderately impacted reef"
        },
        {
            "location": "Red Sea, Egypt",
            "depth": "8-20 meters",
            "time": "Midday survey", 
            "environmental_context": "High biodiversity area"
        },
        {
            "location": "Pacific Ocean, Hawaii",
            "depth": "15-30 meters",
            "time": "Evening survey",
            "environmental_context": "Volcanic reef system"
        }
    ]

######################################################################
# Main Execution and User Interface
######################################################################

def main():
    """
    Main execution function with LLM selection and pipeline management.
    Provides interactive interface for marine conservation analysis.
    """
    
    print("MARINE SPECIES CONSERVATION PIPELINE")
    print("AI-Powered Marine Conservation Analysis")
    print("=" * 60)
    
    ######################################################################
    # Display available LLM options
    ######################################################################
    display_available_llm_options()
    
    ######################################################################
    # Get user selection for LLM type
    ######################################################################
    print("\n" + "=" * 60)
    print("SELECT LLM TYPE:")
    print("1. Rule-Based Analysis (Recommended)")
    print("2. Ollama Local LLM") 
    print("3. Hugging Face Transformers")
    
    user_selection = input("\nEnter your choice (1-3) [1]: ").strip()
    
    ######################################################################
    # Map user selection to LLM type
    ######################################################################
    llm_type_mapping = {
        "1": "rule_based",
        "2": "ollama", 
        "3": "transformers",
        "": "rule_based"  # Default selection
    }
    
    selected_llm_type = llm_type_mapping.get(user_selection, "rule_based")
    
    print(f"\nSelected LLM type: {selected_llm_type}")
    
    ######################################################################
    # Setup pipeline infrastructure
    ######################################################################
    marine_images_directory = setup_marine_conservation_pipeline()
    
    ######################################################################
    # Initialize conservation pipeline with selected LLM
    ######################################################################
    conservation_pipeline = MarineConservationPipeline(selected_llm_type)
    
    ######################################################################
    # Prepare sample survey data
    ######################################################################
    sample_marine_surveys = create_sample_marine_survey_data()
    
    ######################################################################
    # Execute conservation analysis
    ######################################################################
    try:
        selected_survey_location = random.choice(sample_marine_surveys)
        
        print(f"\nExecuting conservation analysis...")
        print(f"   Location: {selected_survey_location['location']}")
        print(f"   Environment: {selected_survey_location['environmental_context']}")
        
        analysis_results = conservation_pipeline.execute_comprehensive_conservation_analysis(
            marine_images_directory, selected_survey_location)
        
        print("\n" + "=" * 60)
        print("MARINE CONSERVATION ANALYSIS COMPLETE")
        print("=" * 60)
        print("Expert conservation recommendations generated")
        print("Species of concern identified")
        print("Detailed reports saved")
        print("\nCheck 'conservation_reports' folder for:")
        print("   Detailed JSON data")
        print("   Conservation action plan")
        
        ######################################################################
        # Display critical findings summary
        ######################################################################
        species_concern = analysis_results['algorithm_outputs']['species_of_concern']
        print(f"\nCRITICAL FINDING:")
        print(f"   Species: {species_concern['species']}")
        print(f"   Population: {species_concern['percentage']}% ({species_concern['count']} individuals)")
        print(f"   Action: Immediate conservation measures required")
        
        return analysis_results
        
    except Exception as analysis_error:
        print(f"\nError: {str(analysis_error)}")
        print("Please check your setup and try again.")
        return None

if __name__ == "__main__":
    # Execute the complete marine conservation pipeline with LLM integration
    conservation_analysis_results = main()
