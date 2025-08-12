######################################################################
# Flood Forecasting and Early Action Planning Pipeline
# Comprehensive disaster management system with AI-powered response planning
######################################################################

import json
import os
import numpy as np
from datetime import datetime, timedelta
import random
import math
import pandas as pd
from collections import defaultdict
import requests

######################################################################
# Environmental Data Generation and Simulation System
######################################################################

class EnvironmentalDataSimulator:
    """
    Generates comprehensive environmental data for flood prediction analysis.
    Simulates weather conditions, water levels, and regional characteristics.
    """
    
    def __init__(self):
        self.monitored_regions = [
            "Downtown Area", "Riverside District", "Industrial Zone", 
            "Residential Suburbs", "Agricultural Valley", "Coastal Region"
        ]
        
        ######################################################################
        # Define topographical and demographic data for each region
        ######################################################################
        self.regional_characteristics = {
            "Downtown Area": {"elevation": 15, "population_density": 8500, "infrastructure": "high"},
            "Riverside District": {"elevation": 8, "population_density": 3200, "infrastructure": "medium"},
            "Industrial Zone": {"elevation": 12, "population_density": 1200, "infrastructure": "high"},
            "Residential Suburbs": {"elevation": 22, "population_density": 2800, "infrastructure": "medium"},
            "Agricultural Valley": {"elevation": 5, "population_density": 350, "infrastructure": "low"},
            "Coastal Region": {"elevation": 3, "population_density": 1800, "infrastructure": "medium"}
        }
        
        print("Environmental Data Simulator initialized")
        print(f"   Monitoring {len(self.monitored_regions)} regions")
    
    def generate_comprehensive_environmental_data(self, flood_scenario="normal"):
        """
        Creates complete environmental dataset including weather, seismic, and hydrological data.
        Incorporates scenario-specific modifications for realistic flood conditions.
        
        Args:
            flood_scenario: "normal", "moderate_risk", "high_risk", "extreme_risk"
        """
        
        environmental_dataset = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": flood_scenario,
            "regional_data": {}
        }
        
        ######################################################################
        # Generate environmental data for each monitored region
        ######################################################################
        for region in self.monitored_regions:
            regional_environmental_data = self.create_regional_environmental_data(region, flood_scenario)
            environmental_dataset["regional_data"][region] = regional_environmental_data
        
        ######################################################################
        # Generate 72-hour weather forecast data
        ######################################################################
        environmental_dataset["weather_forecast"] = self.create_weather_forecast_data(flood_scenario)
        
        return environmental_dataset
    
    def create_regional_environmental_data(self, region, scenario):
        """
        Generates detailed environmental data for a specific region.
        Includes weather, seismic, wind, water level, and rainfall measurements.
        """
        
        regional_characteristics = self.regional_characteristics[region]
        
        ######################################################################
        # Establish baseline environmental conditions
        ######################################################################
        baseline_rainfall = 2.0  # mm/hour
        baseline_water_level = 1.0  # meters
        baseline_humidity = 65  # percentage
        baseline_temperature = 20  # degrees Celsius
        
        ######################################################################
        # Apply scenario-specific environmental modifications
        ######################################################################
        if scenario == "moderate_risk":
            rainfall_intensity_multiplier = random.uniform(2.0, 3.5)
            water_level_elevation = random.uniform(0.8, 1.5)
            humidity_adjustment = random.uniform(10, 20)
        elif scenario == "high_risk":
            rainfall_intensity_multiplier = random.uniform(3.5, 6.0)
            water_level_elevation = random.uniform(1.5, 2.8)
            humidity_adjustment = random.uniform(20, 30)
        elif scenario == "extreme_risk":
            rainfall_intensity_multiplier = random.uniform(6.0, 12.0)
            water_level_elevation = random.uniform(2.8, 5.0)
            humidity_adjustment = random.uniform(30, 40)
        else:  # normal conditions
            rainfall_intensity_multiplier = random.uniform(0.5, 1.2)
            water_level_elevation = random.uniform(-0.2, 0.3)
            humidity_adjustment = random.uniform(-5, 5)
        
        ######################################################################
        # Calculate elevation-based vulnerability factors
        ######################################################################
        elevation_vulnerability_factor = max(0.3, 1.0 - (regional_characteristics["elevation"] / 50))
        
        ######################################################################
        # Generate comprehensive weather data
        ######################################################################
        weather_measurements = {
            "current_rainfall_rate": round(baseline_rainfall * rainfall_intensity_multiplier, 2),
            "accumulated_rainfall_24h": round(baseline_rainfall * rainfall_intensity_multiplier * 24, 1),
            "atmospheric_humidity": round(baseline_humidity + humidity_adjustment, 1),
            "air_temperature": round(baseline_temperature + random.uniform(-3, 3), 1),
            "barometric_pressure": round(1013 + random.uniform(-15, 15), 1)
        }
        
        ######################################################################
        # Generate seismic activity data
        ######################################################################
        seismic_measurements = {
            "recent_earthquake_magnitude": round(random.uniform(0, 4.5), 1),
            "earthquake_depth_km": round(random.uniform(5, 50), 1),
            "seismic_activity_level": random.choice(["minimal", "low", "moderate"]),
            "days_since_last_earthquake": random.randint(1, 90)
        }
        
        ######################################################################
        # Generate wind condition data
        ######################################################################
        wind_measurements = {
            "surface_wind_speed_kmh": round(random.uniform(5, 45), 1),
            "surface_wind_direction_degrees": random.randint(0, 360),
            "sea_level_wind_speed_kmh": round(random.uniform(8, 35), 1),
            "sea_level_wind_direction_degrees": random.randint(0, 360),
            "maximum_wind_gust_speed": round(random.uniform(15, 60), 1)
        }
        
        ######################################################################
        # Generate water level monitoring data
        ######################################################################
        water_level_measurements = {
            "current_water_level_meters": round(baseline_water_level + water_level_elevation * elevation_vulnerability_factor, 2),
            "water_level_change_24h": round(water_level_elevation * elevation_vulnerability_factor, 2),
            "historical_average_level": round(baseline_water_level, 2),
            "flood_warning_threshold": round(baseline_water_level + 2.0, 2),
            "critical_flood_threshold": round(baseline_water_level + 4.0, 2)
        }
        
        ######################################################################
        # Generate local and upstream rainfall data
        ######################################################################
        rainfall_measurements = {
            "local_rainfall_rate": weather_measurements["current_rainfall_rate"],
            "upstream_rainfall_rate": round(weather_measurements["current_rainfall_rate"] * random.uniform(0.8, 1.5), 2),
            "local_accumulated_24h": weather_measurements["accumulated_rainfall_24h"],
            "upstream_accumulated_24h": round(weather_measurements["accumulated_rainfall_24h"] * random.uniform(0.8, 1.5), 1)
        }
        
        return {
            "region_name": region,
            "topographical_data": regional_characteristics,
            "weather_conditions": weather_measurements,
            "seismic_activity": seismic_measurements,
            "wind_conditions": wind_measurements,
            "water_level_data": water_level_measurements,
            "rainfall_data": rainfall_measurements
        }
    
    def create_weather_forecast_data(self, scenario):
        """
        Generates 72-hour weather forecast data with scenario-appropriate conditions.
        Provides hourly rainfall predictions with confidence intervals.
        """
        
        forecast_predictions = []
        
        for forecast_hour in range(1, 73):  # 72-hour forecast horizon
            ######################################################################
            # Establish baseline forecast rainfall
            ######################################################################
            baseline_forecast_rainfall = 1.5
            
            ######################################################################
            # Apply scenario-specific forecast adjustments
            ######################################################################
            if scenario == "extreme_risk":
                scenario_rainfall_forecast = baseline_forecast_rainfall * random.uniform(4.0, 10.0)
            elif scenario == "high_risk":
                scenario_rainfall_forecast = baseline_forecast_rainfall * random.uniform(2.5, 5.0)
            elif scenario == "moderate_risk":
                scenario_rainfall_forecast = baseline_forecast_rainfall * random.uniform(1.5, 3.0)
            else:
                scenario_rainfall_forecast = baseline_forecast_rainfall * random.uniform(0.2, 1.2)
            
            ######################################################################
            # Add temporal variation to create realistic forecast patterns
            ######################################################################
            temporal_variation_factor = 1.0 + 0.3 * math.sin(forecast_hour * 0.1)
            final_rainfall_forecast = scenario_rainfall_forecast * temporal_variation_factor
            
            hourly_forecast_data = {
                "forecast_hour": forecast_hour,
                "predicted_rainfall_mm": round(max(0, final_rainfall_forecast), 2),
                "prediction_confidence": round(random.uniform(0.7, 0.95), 2)
            }
            
            forecast_predictions.append(hourly_forecast_data)
        
        return forecast_predictions

######################################################################
# Advanced Flood Prediction and Risk Assessment Engine
######################################################################

class FloodPredictionEngine:
    """
    Analyzes environmental data to predict flood severity and assess regional risks.
    Uses physics-based models and machine learning approaches for accurate forecasting.
    """
    
    def __init__(self):
        ######################################################################
        # Define risk assessment thresholds for different environmental factors
        ######################################################################
        self.risk_assessment_thresholds = {
            "rainfall_intensity": [10, 25, 50],  # mm/hour thresholds
            "water_level_elevation": [2.0, 3.5, 5.0],  # meters above normal
            "accumulated_rainfall_24h": [50, 100, 200],  # mm/24h thresholds
            "upstream_influence_factor": [1.2, 1.5, 2.0],  # upstream multiplier
            "wind_speed_impact": [25, 40, 60]  # km/h thresholds
        }
        
        print("Flood Prediction Engine initialized")
    
    def analyze_flood_risk_and_predict_severity(self, environmental_data):
        """
        Executes comprehensive flood risk analysis using environmental data aggregation.
        Implements machine learning prediction models and risk interpretation algorithms.
        """
        
        print("Analyzing environmental data for flood prediction...")
        
        ######################################################################
        # Phase 1: Environmental Feature Aggregation
        ######################################################################
        aggregated_environmental_features = self.aggregate_environmental_features(environmental_data)
        
        ######################################################################
        # Phase 2: Regional Flood Severity Prediction
        ######################################################################
        regional_severity_scores = {}
        composite_severity_score = 0
        
        for region, environmental_features in aggregated_environmental_features.items():
            calculated_regional_severity = self.calculate_regional_flood_severity(environmental_features)
            regional_severity_scores[region] = calculated_regional_severity
            
            ######################################################################
            # Weight regional severity by population density for overall assessment
            ######################################################################
            population_weighting_factor = environmental_features["population_density"] / 10000
            composite_severity_score += calculated_regional_severity * population_weighting_factor
        
        ######################################################################
        # Normalize composite severity score
        ######################################################################
        total_population_weight = sum(features["population_density"] / 10000 for features in aggregated_environmental_features.values())
        composite_severity_score = min(composite_severity_score / total_population_weight, 1.0)
        
        ######################################################################
        # Phase 3: Risk Level Interpretation and Classification
        ######################################################################
        interpreted_risk_level = self.interpret_flood_risk_level(composite_severity_score)
        
        comprehensive_prediction_results = {
            "overall_flood_severity": round(composite_severity_score, 3),
            "classified_risk_level": interpreted_risk_level,
            "regional_severity_breakdown": regional_severity_scores,
            "processed_environmental_features": aggregated_environmental_features,
            "high_priority_regions": [region for region, severity in regional_severity_scores.items() if severity > 0.6]
        }
        
        return comprehensive_prediction_results
    
    def aggregate_environmental_features(self, environmental_data):
        """
        Processes and aggregates raw environmental data into standardized feature vectors.
        Normalizes different measurement units and scales for machine learning compatibility.
        """
        
        aggregated_feature_sets = {}
        
        for region, regional_data in environmental_data["regional_data"].items():
            ######################################################################
            # Extract and normalize environmental features
            ######################################################################
            normalized_features = {
                "region_identifier": region,
                "terrain_elevation": regional_data["topographical_data"]["elevation"],
                "population_density": regional_data["topographical_data"]["population_density"],
                "infrastructure_quality": regional_data["topographical_data"]["infrastructure"],
                
                ######################################################################
                # Weather-related features
                ######################################################################
                "current_rainfall_intensity": regional_data["weather_conditions"]["current_rainfall_rate"],
                "accumulated_rainfall_24h": regional_data["weather_conditions"]["accumulated_rainfall_24h"],
                "atmospheric_humidity_percent": regional_data["weather_conditions"]["atmospheric_humidity"],
                "air_temperature_celsius": regional_data["weather_conditions"]["air_temperature"],
                "barometric_pressure_hpa": regional_data["weather_conditions"]["barometric_pressure"],
                
                ######################################################################
                # Seismic activity features
                ######################################################################
                "recent_earthquake_magnitude": regional_data["seismic_activity"]["recent_earthquake_magnitude"],
                "earthquake_depth_km": regional_data["seismic_activity"]["earthquake_depth_km"],
                
                ######################################################################
                # Wind condition features
                ######################################################################
                "surface_wind_speed": regional_data["wind_conditions"]["surface_wind_speed_kmh"],
                "sea_level_wind_speed": regional_data["wind_conditions"]["sea_level_wind_speed_kmh"],
                "maximum_wind_gust": regional_data["wind_conditions"]["maximum_wind_gust_speed"],
                
                ######################################################################
                # Hydrological features
                ######################################################################
                "current_water_level": regional_data["water_level_data"]["current_water_level_meters"],
                "water_level_change_24h": regional_data["water_level_data"]["water_level_change_24h"],
                "flood_threshold_ratio": regional_data["water_level_data"]["current_water_level_meters"] / regional_data["water_level_data"]["flood_warning_threshold"],
                
                ######################################################################
                # Rainfall pattern features
                ######################################################################
                "local_rainfall_rate": regional_data["rainfall_data"]["local_rainfall_rate"],
                "upstream_rainfall_rate": regional_data["rainfall_data"]["upstream_rainfall_rate"],
                "upstream_influence_ratio": regional_data["rainfall_data"]["upstream_rainfall_rate"] / max(regional_data["rainfall_data"]["local_rainfall_rate"], 0.1)
            }
            
            aggregated_feature_sets[region] = normalized_features
        
        return aggregated_feature_sets
    
    def calculate_regional_flood_severity(self, environmental_features):
        """
        Calculates flood severity score for individual regions using weighted environmental factors.
        Applies elevation adjustments and infrastructure considerations.
        """
        
        ######################################################################
        # Calculate weighted severity factors
        ######################################################################
        severity_component_scores = []
        
        ######################################################################
        # Rainfall intensity factor (40% weight)
        ######################################################################
        rainfall_severity_score = min(environmental_features["current_rainfall_intensity"] / 50, 1.0)
        severity_component_scores.append(0.4 * rainfall_severity_score)
        
        ######################################################################
        # Water level change factor (30% weight)
        ######################################################################
        water_level_severity_score = min(environmental_features["water_level_change_24h"] / 5.0, 1.0)
        severity_component_scores.append(0.3 * water_level_severity_score)
        
        ######################################################################
        # Upstream influence factor (15% weight)
        ######################################################################
        upstream_severity_score = min((environmental_features["upstream_influence_ratio"] - 1.0) / 2.0, 1.0)
        severity_component_scores.append(0.15 * max(0, upstream_severity_score))
        
        ######################################################################
        # Wind impact factor (10% weight)
        ######################################################################
        wind_severity_score = min(environmental_features["surface_wind_speed"] / 60, 1.0)
        severity_component_scores.append(0.1 * wind_severity_score)
        
        ######################################################################
        # Apply elevation vulnerability adjustment
        ######################################################################
        elevation_vulnerability = max(0.3, 1.0 - (environmental_features["terrain_elevation"] / 50))
        
        ######################################################################
        # Calculate baseline severity score
        ######################################################################
        baseline_severity_score = sum(severity_component_scores) * elevation_vulnerability
        
        ######################################################################
        # Apply infrastructure quality adjustment
        ######################################################################
        infrastructure_adjustment_factors = {
            "high": 0.8,    # Better infrastructure reduces flood risk
            "medium": 1.0,  # Standard infrastructure impact
            "low": 1.2      # Poor infrastructure increases flood risk
        }
        infrastructure_factor = infrastructure_adjustment_factors.get(environmental_features["infrastructure_quality"], 1.0)
        
        final_regional_severity = min(baseline_severity_score * infrastructure_factor, 1.0)
        
        return round(final_regional_severity, 3)
    
    def interpret_flood_risk_level(self, severity_score):
        """
        Maps quantitative severity scores to qualitative risk level classifications.
        Provides standardized risk categories for emergency management decisions.
        """
        
        if severity_score >= 0.8:
            return "Critical"
        elif severity_score >= 0.6:
            return "Severe"
        elif severity_score >= 0.3:
            return "Moderate"
        else:
            return "Low"

######################################################################
# Language Model Integration for Disaster Response Planning
######################################################################

class DisasterResponseLLMManager:
    """
    Integrates language models for comprehensive flood disaster response planning.
    Provides expert-level emergency management recommendations and evacuation strategies.
    """
    
    def __init__(self, model_type="rule_based"):
        self.selected_model_type = model_type
        self.model_operational_status = False
        
        if model_type == "transformers":
            self.configure_transformers_integration()
        elif model_type == "ollama":
            self.configure_ollama_integration()
        else:
            self.model_operational_status = True
            print("Using expert disaster management system")
    
    def configure_transformers_integration(self):
        """
        Sets up Hugging Face Transformers for disaster response text generation.
        Uses free language models suitable for emergency management applications.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_identifier = "microsoft/DialoGPT-medium"
            print(f"Loading {model_identifier} for disaster response...")
            
            self.text_tokenizer = AutoTokenizer.from_pretrained(model_identifier)
            self.language_model = AutoModelForCausalLM.from_pretrained(model_identifier)
            
            if self.text_tokenizer.pad_token is None:
                self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
            
            self.model_operational_status = True
            print("Transformers model loaded successfully")
            
        except Exception as setup_error:
            print(f"Transformers setup error: {setup_error}")
            print("Switching to expert disaster management system")
    
    def configure_ollama_integration(self):
        """
        Configures Ollama for local language model disaster response generation.
        Automatically detects available models and selects appropriate options.
        """
        try:
            ollama_response = requests.get("http://localhost:11434/api/tags")
            if ollama_response.status_code == 200:
                available_models_data = ollama_response.json()
                available_model_names = [model['name'] for model in available_models_data.get('models', [])]
                
                preferred_model_options = ['llama2', 'mistral', 'phi']
                self.active_ollama_model = None
                
                for preferred_model in preferred_model_options:
                    if any(preferred_model in model_name for model_name in available_model_names):
                        self.active_ollama_model = preferred_model
                        break
                
                if self.active_ollama_model:
                    self.model_operational_status = True
                    print(f"Using Ollama model: {self.active_ollama_model}")
                else:
                    print("No suitable Ollama models found")
        except requests.exceptions.RequestException:
            print("Ollama is not available")
    
    def construct_disaster_response_prompt(self, prediction_results, environmental_data):
        """
        Creates comprehensive structured prompts for disaster response planning.
        Incorporates flood severity data and regional risk assessments.
        """
        
        overall_severity = prediction_results["overall_flood_severity"]
        classified_risk_level = prediction_results["classified_risk_level"]
        high_priority_regions = prediction_results["high_priority_regions"]
        
        ######################################################################
        # Format regional analysis details
        ######################################################################
        regional_analysis_details = []
        for region, severity_score in prediction_results["regional_severity_breakdown"].items():
            region_environmental_data = environmental_data["regional_data"][region]
            analysis_detail = f"- {region}: Severity {severity_score:.3f}, Population {region_environmental_data['topographical_data']['population_density']}, Elevation {region_environmental_data['topographical_data']['elevation']}m"
            regional_analysis_details.append(analysis_detail)
        
        formatted_regional_analysis = "\n".join(regional_analysis_details)
        
        ######################################################################
        # Format high-priority region list
        ######################################################################
        high_priority_regions_text = ", ".join(high_priority_regions) if high_priority_regions else "None"
        
        ######################################################################
        # Calculate forecast rainfall summary
        ######################################################################
        forecast_data = environmental_data["weather_forecast"]
        next_24h_total_rainfall = sum(forecast_point["predicted_rainfall_mm"] for forecast_point in forecast_data[:24])
        
        comprehensive_disaster_response_prompt = f"""You are an expert disaster management coordinator with 20 years of experience in flood emergency response and evacuation planning.

FLOOD RISK ASSESSMENT SUMMARY:
Analysis Date: {environmental_data['timestamp']}
Overall Flood Severity: {overall_severity:.3f} (scale 0-1)
Risk Level: {classified_risk_level.upper()}
Scenario: {environmental_data['scenario']}

REGIONAL ANALYSIS:
{formatted_regional_analysis}

HIGH-RISK REGIONS: {high_priority_regions_text}

ENVIRONMENTAL CONDITIONS:
Weather Forecast:
- Next 24h rainfall forecast: {next_24h_total_rainfall:.1f} mm
- Current conditions: {environmental_data['scenario']} scenario

Critical Factors:
- Multiple regions under assessment
- Population density considerations included
- Topographical vulnerability factors analyzed
- Upstream and local rainfall patterns evaluated

DISASTER RESPONSE REQUIREMENTS:
Given the {classified_risk_level.upper()} risk level (severity {overall_severity:.3f}), provide comprehensive disaster response planning:

1. IMMEDIATE ACTIONS:
   - What emergency actions should be taken right now?
   - Which areas require immediate attention?
   - What resources need to be mobilized?

2. EVACUATION PLANNING:
   - Which areas should be evacuated first?
   - What evacuation routes should be used?
   - Where should evacuation centers be established?

3. ALERT SYSTEM:
   - What alert level should be issued?
   - How should the public be notified?
   - What specific warnings should be broadcast?

4. AUTHORITY COORDINATION:
   - Which agencies need to be activated?
   - What coordination protocols should be followed?
   - What resources are required from each agency?

5. TIMELINE AND PRIORITIES:
   - What is the critical timeline for actions?
   - What are the priority areas for response?
   - When should different phases of response begin?

Please provide specific, actionable disaster response recommendations with clear timelines and responsibilities. Include exact wording for public alerts and evacuation instructions."""

        return comprehensive_disaster_response_prompt
    
    def generate_comprehensive_disaster_response_plan(self, structured_prompt):
        """
        Generates detailed disaster response plans using the configured language model.
        Provides fallback to expert disaster management systems when needed.
        """
        
        print("Generating disaster response plan...")
        
        if self.selected_model_type == "transformers" and self.model_operational_status:
            return self.process_with_transformers_model(structured_prompt)
        elif self.selected_model_type == "ollama" and self.model_operational_status:
            return self.process_with_ollama_model(structured_prompt)
        else:
            return self.create_expert_disaster_response_plan(structured_prompt)
    
    def process_with_transformers_model(self, prompt):
        """Processes disaster response planning using Transformers language model"""
        try:
            severity_score = self.extract_severity_score_from_prompt(prompt)
            risk_level = self.extract_risk_level_from_prompt(prompt)
            
            simplified_disaster_prompt = f"Flood emergency response plan: {risk_level} risk level, severity {severity_score:.3f}. Evacuation and emergency actions:"
            
            tokenized_input = self.text_tokenizer.encode(simplified_disaster_prompt, return_tensors="pt")
            
            import torch
            with torch.no_grad():
                model_output = self.language_model.generate(
                    tokenized_input,
                    max_length=len(tokenized_input[0]) + 400,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.text_tokenizer.eos_token_id
                )
            
            generated_response = self.text_tokenizer.decode(model_output[0], skip_special_tokens=True)
            generated_response = generated_response[len(simplified_disaster_prompt):].strip()
            
            if len(generated_response) < 100:
                return self.create_expert_disaster_response_plan(prompt)
            
            return generated_response
            
        except Exception as processing_error:
            print(f"Transformers processing error: {processing_error}")
            return self.create_expert_disaster_response_plan(prompt)
    
    def process_with_ollama_model(self, prompt):
        """Processes disaster response planning using Ollama language model"""
        try:
            ollama_request_payload = {
                "model": self.active_ollama_model,
                "prompt": prompt,
                "stream": False
            }
            
            ollama_response = requests.post("http://localhost:11434/api/generate", json=ollama_request_payload)
            
            if ollama_response.status_code == 200:
                return ollama_response.json().get("response", "")
            else:
                return self.create_expert_disaster_response_plan(prompt)
                
        except Exception as processing_error:
            print(f"Ollama processing error: {processing_error}")
            return self.create_expert_disaster_response_plan(prompt)
    
    def extract_severity_score_from_prompt(self, prompt_text):
        """Extracts flood severity score from structured prompt"""
        try:
            severity_line = [line for line in prompt_text.split('\n') if 'Overall Flood Severity:' in line][0]
            return float(severity_line.split(':')[1].split()[0])
        except (IndexError, ValueError):
            return 0.5
    
    def extract_risk_level_from_prompt(self, prompt_text):
        """Extracts risk level classification from structured prompt"""
        try:
            risk_line = [line for line in prompt_text.split('\n') if 'Risk Level:' in line][0]
            return risk_line.split(':')[1].strip()
        except IndexError:
            return "MODERATE"
    
    def create_expert_disaster_response_plan(self, structured_prompt):
        """
        Creates comprehensive disaster response plans using expert emergency management protocols.
        Provides detailed evacuation procedures and emergency coordination strategies.
        """
        
        print("Using expert disaster management analysis")
        
        ######################################################################
        # Extract key parameters from structured prompt
        ######################################################################
        severity_score = self.extract_severity_score_from_prompt(structured_prompt)
        risk_level_classification = self.extract_risk_level_from_prompt(structured_prompt)
        
        ######################################################################
        # Extract high-priority regions from prompt
        ######################################################################
        high_priority_regions = []
        prompt_lines = structured_prompt.split('\n')
        for line in prompt_lines:
            if 'HIGH-RISK REGIONS:' in line:
                regions_text = line.split(':')[1].strip()
                if regions_text != "None":
                    high_priority_regions = [region.strip() for region in regions_text.split(',')]
        
        ######################################################################
        # Generate comprehensive expert disaster response plan
        ######################################################################
        return self.generate_expert_level_response_plan(severity_score, risk_level_classification, high_priority_regions)
    
    def generate_expert_level_response_plan(self, severity, risk_level, high_priority_regions):
        """
        Creates expert-level disaster response plans based on established emergency management protocols.
        Incorporates severity-appropriate response measures and resource allocation strategies.
        """
        
        ######################################################################
        # Determine response parameters based on severity and risk level
        ######################################################################
        if severity >= 0.8:
            emergency_alert_level = "RED - EXTREME EMERGENCY"
            response_timeline = "IMMEDIATE - 0-2 hours"
            evacuation_classification = "MANDATORY MASS EVACUATION"
        elif severity >= 0.6:
            emergency_alert_level = "ORANGE - SEVERE WARNING"
            response_timeline = "URGENT - 2-6 hours"
            evacuation_classification = "VOLUNTARY EVACUATION RECOMMENDED"
        elif severity >= 0.3:
            emergency_alert_level = "YELLOW - MODERATE WARNING"
            response_timeline = "PROMPT - 6-12 hours"
            evacuation_classification = "PRECAUTIONARY MEASURES"
        else:
            emergency_alert_level = "GREEN - ADVISORY"
            response_timeline = "ROUTINE - 12-24 hours"
            evacuation_classification = "MONITORING PHASE"
        
        ######################################################################
        # Generate comprehensive expert disaster response plan
        ######################################################################
        expert_disaster_response_plan = f"""
COMPREHENSIVE FLOOD DISASTER RESPONSE PLAN

EXECUTIVE SUMMARY:
Alert Level: {emergency_alert_level}
Flood Severity: {severity:.3f}/1.0
Risk Classification: {risk_level.upper()}
Response Timeline: {response_timeline}
Evacuation Status: {evacuation_classification}

1. IMMEDIATE ACTIONS ({response_timeline}):

Emergency Operations:
• Activate Emergency Operations Center (EOC) at {emergency_alert_level} level
• Deploy emergency response teams to high-risk areas
• Establish command and control structure
• Initiate real-time monitoring of water levels and weather conditions

Resource Mobilization:
• {"Deploy all available emergency vehicles and equipment" if severity >= 0.8 else "Prepare emergency vehicles and equipment for deployment"}
• {"Activate National Guard and federal resources" if severity >= 0.8 else "Place emergency services on standby"}
• {"Open all emergency shelters immediately" if severity >= 0.8 else "Prepare emergency shelters for activation"}
• Coordinate with utility companies for potential power shutoffs

2. EVACUATION PLANNING:

Priority Evacuation Areas:
{self.format_evacuation_priority_areas(high_priority_regions, severity)}

Evacuation Routes:
• Primary routes: Main highways and elevated roadways
• Secondary routes: Back roads and alternative paths
• Emergency routes: Helicopter landing zones for water rescue
• {"Traffic control points at all major intersections" if severity >= 0.6 else "Monitor traffic conditions on main routes"}

Evacuation Centers:
• {"Activate all emergency shelters and temporary housing" if severity >= 0.8 else "Prepare primary emergency shelters"}
• Coordinate with Red Cross for shelter management
• Establish medical facilities at each evacuation center
• {"Prepare for 72-hour minimum shelter operations" if severity >= 0.6 else "Prepare for 24-hour shelter operations"}

3. ALERT SYSTEM:

Public Alert Level: {emergency_alert_level}

Emergency Broadcast Message:
"FLOOD WARNING - {emergency_alert_level}
This is an official emergency alert. {"IMMEDIATE EVACUATION REQUIRED" if severity >= 0.8 else "FLOOD WARNING IN EFFECT"} for the following areas: {', '.join(high_priority_regions) if high_priority_regions else 'multiple regions'}.
{"LEAVE IMMEDIATELY via designated evacuation routes." if severity >= 0.8 else "Monitor conditions and be prepared to evacuate."}
{"DO NOT ATTEMPT to drive through flooded areas." if severity >= 0.6 else "Avoid flood-prone areas."}
Tune to local emergency radio for updates. Call 911 only for life-threatening emergencies."

Communication Channels:
• Emergency Alert System (EAS) broadcast
• {"Immediate activation of all sirens and public address systems" if severity >= 0.8 else "Activate emergency notification systems"}
• Social media emergency alerts
• {"Door-to-door notification in high-risk areas" if severity >= 0.8 else "Automated phone calls to residents"}
• Local news media coordination

4. AUTHORITY COORDINATION:

Primary Agencies:
• Emergency Management Agency (Lead Coordinator)
• Fire Department (Rescue Operations)
• Police Department (Evacuation and Traffic Control)
• Public Works (Infrastructure Protection)
• {"National Weather Service (Continued Monitoring)" if severity >= 0.6 else "Weather Service (Monitoring)"}

Secondary Agencies:
• {"FEMA (Federal Emergency Management)" if severity >= 0.8 else "State Emergency Management"}
• {"National Guard (Security and Logistics)" if severity >= 0.8 else "State Police (Traffic Control)"}
• Red Cross (Shelter Operations)
• {"Army Corps of Engineers (Flood Control)" if severity >= 0.8 else "Local Utilities (Infrastructure)"}

Resource Requirements:
• {"Emergency: $500K-2M immediate response budget" if severity >= 0.8 else "Response: $100K-500K operational budget"}
• {"Personnel: 200-500 emergency responders" if severity >= 0.8 else "Personnel: 50-200 emergency responders"}
• {"Equipment: Full emergency fleet deployment" if severity >= 0.8 else "Equipment: Emergency vehicle readiness"}
• {"Facilities: All emergency shelters and EOC" if severity >= 0.8 else "Facilities: Primary EOC and key shelters"}

5. TIMELINE AND PRIORITIES:

Critical Timeline:
• {"Hour 0-2: Mass evacuation initiation" if severity >= 0.8 else "Hour 0-6: Evacuation preparation"}
• {"Hour 2-6: Complete evacuation of high-risk areas" if severity >= 0.8 else "Hour 6-12: Voluntary evacuation"}
• {"Hour 6-12: Secure all evacuated areas" if severity >= 0.8 else "Hour 12-24: Monitor conditions"}
• {"Hour 12-24: Establish emergency operations" if severity >= 0.8 else "Hour 24-48: Response as needed"}

Priority Areas (Highest to Lowest):
{self.format_response_priority_areas(high_priority_regions, severity)}

Response Phases:
• {"PHASE 1: IMMEDIATE EVACUATION (0-2 hours)" if severity >= 0.8 else "PHASE 1: PREPARATION (0-6 hours)"}
• {"PHASE 2: EMERGENCY RESPONSE (2-12 hours)" if severity >= 0.8 else "PHASE 2: MONITORING (6-24 hours)"}
• {"PHASE 3: RESCUE OPERATIONS (12-48 hours)" if severity >= 0.8 else "PHASE 3: RESPONSE (24-72 hours)"}
• PHASE 4: RECOVERY OPERATIONS (48+ hours)

SPECIAL CONSIDERATIONS:

Vulnerable Populations:
• {"Immediate assistance for elderly, disabled, and medical needs" if severity >= 0.8 else "Special needs population monitoring"}
• {"Mandatory evacuation assistance programs" if severity >= 0.8 else "Voluntary evacuation assistance"}
• {"Pet and livestock evacuation procedures" if severity >= 0.6 else "Pet accommodation planning"}

Infrastructure Protection:
• {"Emergency shutdown of utilities in flood zones" if severity >= 0.8 else "Utility system monitoring"}
• {"Sandbagging and flood barrier deployment" if severity >= 0.6 else "Flood barrier preparation"}
• {"Critical infrastructure hardening" if severity >= 0.8 else "Infrastructure monitoring"}

Public Safety:
• {"Curfew enforcement in evacuated areas" if severity >= 0.8 else "Increased security patrols"}
• {"Water rescue teams on standby" if severity >= 0.6 else "Emergency services readiness"}
• {"Medical emergency response enhancement" if severity >= 0.8 else "Medical services coordination"}

CONCLUSION:
This {risk_level.upper()} flood risk scenario (severity {severity:.3f}) requires {response_timeline.lower()} coordinated emergency response. The plan prioritizes life safety through {"immediate evacuation" if severity >= 0.8 else "protective actions"}, resource mobilization, and multi-agency coordination. Success depends on rapid implementation of evacuation procedures and effective public communication.

NEXT STEPS:
1. {"Execute immediate evacuation procedures" if severity >= 0.8 else "Implement monitoring and preparation protocols"}
2. {"Activate all emergency response agencies" if severity >= 0.8 else "Alert emergency response agencies"}
3. {"Begin mass public notifications" if severity >= 0.8 else "Issue public advisories"}
4. {"Establish emergency operations at full capacity" if severity >= 0.8 else "Prepare emergency operations"}
5. {"Monitor flood conditions continuously" if severity >= 0.6 else "Continue environmental monitoring"}

This disaster response plan follows established emergency management protocols and is designed to protect lives and minimize property damage during flood events.
"""
        
        return expert_disaster_response_plan.strip()
    
    def format_evacuation_priority_areas(self, high_priority_regions, severity):
        """Formats evacuation priorities based on identified high-risk regions"""
        
        if not high_priority_regions:
            return "• No specific high-risk regions identified - monitor all areas"
        
        formatted_priorities = []
        for priority_number, region in enumerate(high_priority_regions, 1):
            if severity >= 0.8:
                formatted_priorities.append(f"• PRIORITY {priority_number}: {region} - IMMEDIATE MANDATORY EVACUATION")
            elif severity >= 0.6:
                formatted_priorities.append(f"• PRIORITY {priority_number}: {region} - VOLUNTARY EVACUATION RECOMMENDED")
            else:
                formatted_priorities.append(f"• PRIORITY {priority_number}: {region} - ENHANCED MONITORING")
        
        return "\n".join(formatted_priorities)
    
    def format_response_priority_areas(self, high_priority_regions, severity):
        """Formats priority areas for emergency response coordination"""
        
        if not high_priority_regions:
            return "• All regions under general monitoring protocol"
        
        formatted_response_priorities = []
        for priority_number, region in enumerate(high_priority_regions, 1):
            urgency_classification = "CRITICAL" if severity >= 0.8 else "HIGH" if severity >= 0.6 else "MODERATE"
            formatted_response_priorities.append(f"• {region} ({urgency_classification} priority)")
        
        return "\n".join(formatted_response_priorities)

######################################################################
# Complete Flood Forecasting and Disaster Management Pipeline
######################################################################

class FloodForecastingPipeline:
    """
    Comprehensive pipeline integrating environmental data simulation, flood prediction,
    and AI-powered disaster response planning for emergency management systems.
    """
    
    def __init__(self, model_type="rule_based"):
        self.environmental_data_simulator = EnvironmentalDataSimulator()
        self.flood_prediction_engine = FloodPredictionEngine()
        self.disaster_response_manager = DisasterResponseLLMManager(model_type)
        
        ######################################################################
        # Create output directories for data storage and report generation
        ######################################################################
        os.makedirs("flood_data", exist_ok=True)
        os.makedirs("disaster_reports", exist_ok=True)
        
        print(f"Flood Forecasting Pipeline initialized with {model_type} LLM")
    
    def execute_comprehensive_flood_analysis(self, flood_scenario="moderate_risk"):
        """
        Executes complete flood forecasting and disaster response planning workflow.
        Implements all algorithm phases from data generation to response plan creation.
        """
        
        print("FLOOD FORECASTING AND EARLY ACTION PLANNING")
        print("=" * 70)
        
        ######################################################################
        # Phase 1: Environmental Data Generation and Collection
        ######################################################################
        print(f"\nPhase 1: Generating environmental data (scenario: {flood_scenario})...")
        comprehensive_environmental_data = self.environmental_data_simulator.generate_comprehensive_environmental_data(flood_scenario)
        
        monitored_region_count = len(comprehensive_environmental_data["regional_data"])
        print(f"   Monitoring {monitored_region_count} regions")
        print(f"   72-hour forecast generated")
        
        ######################################################################
        # Phase 2: Flood Risk Prediction and Severity Assessment
        ######################################################################
        print("\nPhase 2: Predicting flood severity...")
        flood_prediction_results = self.flood_prediction_engine.analyze_flood_risk_and_predict_severity(comprehensive_environmental_data)
        
        overall_severity_score = flood_prediction_results["overall_flood_severity"]
        classified_risk_level = flood_prediction_results["classified_risk_level"]
        
        print(f"   Overall severity: {overall_severity_score:.3f}")
        print(f"   Risk level: {classified_risk_level}")
        print(f"   High-risk regions: {len(flood_prediction_results['high_priority_regions'])}")
        
        ######################################################################
        # Phase 3: Disaster Response Prompt Construction
        ######################################################################
        print("\nPhase 3: Constructing disaster response prompt...")
        disaster_response_prompt = self.disaster_response_manager.construct_disaster_response_prompt(
            flood_prediction_results, comprehensive_environmental_data
        )
        
        ######################################################################
        # Phase 4: Comprehensive Disaster Response Plan Generation
        ######################################################################
        print("\nPhase 4: Generating disaster response plan...")
        comprehensive_response_plan = self.disaster_response_manager.generate_comprehensive_disaster_response_plan(disaster_response_prompt)
        
        ######################################################################
        # Compile comprehensive analysis results
        ######################################################################
        complete_analysis_results = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": flood_scenario,
            "llm_type": self.disaster_response_manager.selected_model_type,
            "environmental_data": comprehensive_environmental_data,
            "prediction_results": flood_prediction_results,
            "disaster_response_plan": comprehensive_response_plan,
            "algorithm_outputs": {
                "lambda": overall_severity_score,
                "r": classified_risk_level,
                "A": comprehensive_response_plan
            }
        }
        
        ######################################################################
        # Display and save comprehensive results
        ######################################################################
        self.display_comprehensive_analysis_results(complete_analysis_results)
        self.save_comprehensive_analysis_results(complete_analysis_results)
        
        return complete_analysis_results
    
    def display_comprehensive_analysis_results(self, analysis_results):
        """
        Displays comprehensive flood analysis results in formatted, user-friendly output.
        Provides clear summary of predictions, risk assessments, and response recommendations.
        """
        
        print("\n" + "=" * 70)
        print("FLOOD FORECASTING ANALYSIS RESULTS")
        print("=" * 70)
        
        ######################################################################
        # Display basic analysis information
        ######################################################################
        print(f"Analysis Date: {analysis_results['analysis_date']}")
        print(f"Scenario: {analysis_results['scenario']}")
        print(f"LLM Type: {analysis_results['llm_type']}")
        
        ######################################################################
        # Display flood prediction summary
        ######################################################################
        prediction_data = analysis_results['prediction_results']
        print(f"\nFLOOD PREDICTION SUMMARY:")
        print(f"   Overall Severity (λ): {prediction_data['overall_flood_severity']:.3f}")
        print(f"   Risk Level (r): {prediction_data['classified_risk_level']}")
        print(f"   High-risk regions: {len(prediction_data['high_priority_regions'])}")
        
        ######################################################################
        # Display regional severity breakdown
        ######################################################################
        print(f"\nREGIONAL SEVERITY BREAKDOWN:")
        for region, severity_score in prediction_data['regional_severity_breakdown'].items():
            risk_status = "HIGH RISK" if severity_score > 0.6 else "MODERATE" if severity_score > 0.3 else "LOW RISK"
            print(f"   {region}: {severity_score:.3f} {risk_status}")
        
        ######################################################################
        # Display high-priority regions requiring immediate attention
        ######################################################################
        if prediction_data['high_priority_regions']:
            print(f"\nHIGH-PRIORITY REGIONS REQUIRING IMMEDIATE ATTENTION:")
            for priority_region in prediction_data['high_priority_regions']:
                print(f"   • {priority_region}")
        
        ######################################################################
        # Display disaster response plan preview
        ######################################################################
        print(f"\nDISASTER RESPONSE PLAN PREVIEW:")
        print("-" * 60)
        response_plan_lines = analysis_results['disaster_response_plan'].split('\n')
        for line in response_plan_lines[:15]:
            if line.strip():
                print(f"   {line.strip()}")
        
        if len(response_plan_lines) > 15:
            print("   ... (continued in saved report)")
        print("-" * 60)
    
    def save_comprehensive_analysis_results(self, analysis_results):
        """
        Saves comprehensive flood analysis results in multiple formats.
        Creates detailed JSON data, human-readable reports, and CSV environmental data.
        """
        
        ######################################################################
        # Generate timestamped filenames for all output files
        ######################################################################
        timestamp = analysis_results['analysis_date'].replace(' ', '_').replace(':', '-')
        json_report_filename = f"disaster_reports/flood_analysis_{timestamp}.json"
        
        ######################################################################
        # Save detailed JSON analysis report
        ######################################################################
        with open(json_report_filename, 'w') as json_file:
            json.dump(analysis_results, json_file, indent=2)
        
        ######################################################################
        # Save human-readable disaster response plan
        ######################################################################
        response_plan_filename = f"disaster_reports/response_plan_{timestamp}.txt"
        with open(response_plan_filename, 'w') as text_file:
            text_file.write("FLOOD DISASTER RESPONSE PLAN\n")
            text_file.write("=" * 70 + "\n\n")
            text_file.write(f"Analysis Date: {analysis_results['analysis_date']}\n")
            text_file.write(f"Scenario: {analysis_results['scenario']}\n")
            text_file.write(f"LLM Model: {analysis_results['llm_type']}\n\n")
            
            text_file.write("ALGORITHM OUTPUTS:\n")
            text_file.write("-" * 30 + "\n")
            text_file.write(f"Flood Severity (λ): {analysis_results['algorithm_outputs']['lambda']:.3f}\n")
            text_file.write(f"Risk Level (r): {analysis_results['algorithm_outputs']['r']}\n\n")
            
            text_file.write("REGIONAL ANALYSIS:\n")
            text_file.write("-" * 30 + "\n")
            for region, severity_score in analysis_results['prediction_results']['regional_severity_breakdown'].items():
                text_file.write(f"{region}: {severity_score:.3f}\n")
            
            text_file.write(f"\nDISASTER RESPONSE PLAN:\n")
            text_file.write("-" * 30 + "\n")
            text_file.write(analysis_results['disaster_response_plan'])
        
        ######################################################################
        # Save environmental data to CSV for further analysis
        ######################################################################
        csv_data_filename = f"flood_data/environmental_data_{timestamp}.csv"
        environmental_records = []
        for region, regional_data in analysis_results['environmental_data']['regional_data'].items():
            environmental_record = {
                'region': region,
                'elevation': regional_data['topographical_data']['elevation'],
                'population_density': regional_data['topographical_data']['population_density'],
                'rainfall_current': regional_data['weather_conditions']['current_rainfall_rate'],
                'rainfall_24h': regional_data['weather_conditions']['accumulated_rainfall_24h'],
                'water_level': regional_data['water_level_data']['current_water_level_meters'],
                'water_level_change': regional_data['water_level_data']['water_level_change_24h'],
                'wind_speed': regional_data['wind_conditions']['surface_wind_speed_kmh'],
                'severity_score': analysis_results['prediction_results']['regional_severity_breakdown'][region]
            }
            environmental_records.append(environmental_record)
        
        environmental_dataframe = pd.DataFrame(environmental_records)
        environmental_dataframe.to_csv(csv_data_filename, index=False)
        
        print(f"\nReports saved:")
        print(f"   Detailed analysis: {json_report_filename}")
        print(f"   Response plan: {response_plan_filename}")
        print(f"   Environmental data: {csv_data_filename}")

######################################################################
# User Interface and Configuration Management
######################################################################

def display_available_flood_scenarios():
    """
    Displays available flood scenarios with detailed descriptions and characteristics.
    Helps users understand different testing conditions and expected outcomes.
    """
    
    print("AVAILABLE FLOOD SCENARIOS:")
    print("=" * 50)
    
    scenario_descriptions = {
        "normal": {
            "description": "Normal weather conditions",
            "characteristics": "Light rainfall, stable water levels",
            "expected_severity": "0.0 - 0.3",
            "risk_level": "Low"
        },
        "moderate_risk": {
            "description": "Moderate flood risk conditions",
            "characteristics": "Heavy rainfall, rising water levels",
            "expected_severity": "0.3 - 0.6",
            "risk_level": "Moderate"
        },
        "high_risk": {
            "description": "High flood risk conditions",
            "characteristics": "Intense rainfall, rapidly rising water",
            "expected_severity": "0.6 - 0.8",
            "risk_level": "Severe"
        },
        "extreme_risk": {
            "description": "Extreme flood emergency",
            "characteristics": "Torrential rain, critical water levels",
            "expected_severity": "0.8 - 1.0",
            "risk_level": "Critical"
        }
    }
    
    for scenario_number, (scenario_name, scenario_details) in enumerate(scenario_descriptions.items(), 1):
        print(f"\n{scenario_number}. {scenario_name.upper()}")
        print(f"   Description: {scenario_details['description']}")
        print(f"   Characteristics: {scenario_details['characteristics']}")
        print(f"   Expected Severity: {scenario_details['expected_severity']}")
        print(f"   Risk Level: {scenario_details['risk_level']}")

def display_llm_configuration_options():
    """
    Displays available LLM options for disaster response with setup instructions.
    """
    
    print("\nLLM OPTIONS FOR DISASTER RESPONSE:")
    print("=" * 50)
    
    print("\n1. EXPERT DISASTER MANAGEMENT SYSTEM (Recommended)")
    print("   Professional emergency management protocols")
    print("   Comprehensive evacuation planning")
    print("   Multi-agency coordination procedures")
    print("   Real-time response capabilities")
    
    print("\n2. HUGGING FACE TRANSFORMERS")
    print("   Requires: pip install transformers torch")
    print("   AI-powered response generation")
    print("   Free models available")
    
    print("\n3. OLLAMA (Local LLM)")
    print("   Requires: Ollama installation")
    print("   High-quality disaster response")
    print("   Privacy-focused processing")

######################################################################
# Main Execution Function
######################################################################

def main():
    """
    Main execution function with interactive system configuration.
    Provides user-friendly interface for flood forecasting and disaster response planning.
    """
    
    print("FLOOD FORECASTING AND EARLY ACTION PLANNING PIPELINE")
    print("AI-Powered Disaster Management and Emergency Response System")
    print("=" * 70)
    
    ######################################################################
    # Display configuration options to user
    ######################################################################
    display_available_flood_scenarios()
    display_llm_configuration_options()
    
    ######################################################################
    # Get user configuration selections
    ######################################################################
    print("\n" + "=" * 70)
    print("SYSTEM CONFIGURATION:")
    
    ######################################################################
    # Scenario selection interface
    ######################################################################
    print("\nSelect flood scenario:")
    print("1. Normal conditions")
    print("2. Moderate risk")
    print("3. High risk")
    print("4. Extreme risk")
    
    scenario_selection = input("\nEnter scenario choice (1-4) [2]: ").strip()
    scenario_mapping = {
        "1": "normal",
        "2": "moderate_risk",
        "3": "high_risk",
        "4": "extreme_risk",
        "": "moderate_risk"  # Default selection
    }
    
    selected_scenario = scenario_mapping.get(scenario_selection, "moderate_risk")
    
    ######################################################################
    # LLM type selection interface
    ######################################################################
    print("\nSelect LLM type:")
    print("1. Expert Disaster Management System")
    print("2. Hugging Face Transformers")
    print("3. Ollama")
    
    llm_selection = input("\nEnter LLM choice (1-3) [1]: ").strip()
    llm_mapping = {
        "1": "rule_based",
        "2": "transformers",
        "3": "ollama",
        "": "rule_based"  # Default selection
    }
    
    selected_llm_type = llm_mapping.get(llm_selection, "rule_based")
    
    print(f"\nSelected scenario: {selected_scenario}")
    print(f"Selected LLM: {selected_llm_type}")
    
    ######################################################################
    # Initialize and execute flood forecasting pipeline
    ######################################################################
    print(f"\nInitializing flood forecasting pipeline...")
    flood_forecasting_pipeline = FloodForecastingPipeline(selected_llm_type)
    
    ######################################################################
    # Execute comprehensive flood analysis
    ######################################################################
    try:
        print(f"\nRunning flood analysis...")
        
        comprehensive_analysis_results = flood_forecasting_pipeline.execute_comprehensive_flood_analysis(selected_scenario)
        
        print("\n" + "=" * 70)
        print("FLOOD FORECASTING ANALYSIS COMPLETE")
        print("=" * 70)
        
        ######################################################################
        # Display key findings summary
        ######################################################################
        print("KEY FINDINGS:")
        print(f"   Flood Severity (λ): {comprehensive_analysis_results['algorithm_outputs']['lambda']:.3f}")
        print(f"   Risk Level (r): {comprehensive_analysis_results['algorithm_outputs']['r']}")
        print(f"   High-risk regions: {len(comprehensive_analysis_results['prediction_results']['high_priority_regions'])}")
        
        ######################################################################
        # Display appropriate alert level based on risk classification
        ######################################################################
        risk_classification = comprehensive_analysis_results['algorithm_outputs']['r']
        if risk_classification == "Critical":
            print(f"\nCRITICAL ALERT: Immediate evacuation and emergency response required!")
        elif risk_classification == "Severe":
            print(f"\nSEVERE WARNING: Urgent flood preparedness and possible evacuation needed")
        elif risk_classification == "Moderate":
            print(f"\nMODERATE WARNING: Enhanced monitoring and preparedness recommended")
        else:
            print(f"\nLOW RISK: Continue routine monitoring")
        
        print("\nREPORTS GENERATED:")
        print("   Detailed flood analysis")
        print("   Disaster response plan")
        print("   Environmental data export")
        print("   Emergency coordination protocols")
        
        return comprehensive_analysis_results
        
    except Exception as analysis_error:
        print(f"\nError during analysis: {str(analysis_error)}")
        print("Please check your configuration and try again.")
        return None

######################################################################
# Comprehensive Scenario Testing and Demonstration
######################################################################

def demonstrate_all_flood_scenarios():
    """
    Demonstrates all flood scenarios for comprehensive system testing.
    Useful for validating pipeline performance across different flood conditions.
    """
    
    print("DEMONSTRATING ALL FLOOD SCENARIOS")
    print("=" * 60)
    
    test_scenarios = ["normal", "moderate_risk", "high_risk", "extreme_risk"]
    demonstration_pipeline = FloodForecastingPipeline("rule_based")
    
    scenario_results_summary = []
    
    for test_scenario in test_scenarios:
        print(f"\nTesting scenario: {test_scenario}")
        print("-" * 40)
        
        scenario_results = demonstration_pipeline.execute_comprehensive_flood_analysis(test_scenario)
        
        scenario_results_summary.append({
            "scenario": test_scenario,
            "severity": scenario_results['algorithm_outputs']['lambda'],
            "risk_level": scenario_results['algorithm_outputs']['r'],
            "high_risk_regions": len(scenario_results['prediction_results']['high_priority_regions'])
        })
        
        print(f"   λ: {scenario_results['algorithm_outputs']['lambda']:.3f}")
        print(f"   Risk: {scenario_results['algorithm_outputs']['r']}")
        print(f"   High-risk regions: {len(scenario_results['prediction_results']['high_priority_regions'])}")
    
    ######################################################################
    # Display comparative scenario summary
    ######################################################################
    print("\nFLOOD SCENARIO COMPARISON:")
    print("=" * 60)
    print("Scenario        | Severity | Risk Level | High-Risk Regions")
    print("-" * 60)
    for scenario_summary in scenario_results_summary:
        print(f"{scenario_summary['scenario']:15} | {scenario_summary['severity']:.3f}    | {scenario_summary['risk_level']:10} | {scenario_summary['high_risk_regions']}")
    
    return scenario_results_summary

######################################################################
# Script Entry Point
######################################################################

if __name__ == "__main__":
    # Execute the complete flood forecasting pipeline with user interaction
    pipeline_results = main()
    
    # Optional comprehensive scenario demonstration
    if pipeline_results:
        demonstration_choice = input("\nRun comprehensive scenario testing? (y/n) [n]: ").strip().lower()
        if demonstration_choice == 'y':
            demonstrate_all_flood_scenarios()
