# ============================================================================
# FLOOD FORECASTING AND EARLY ACTION PLANNING PIPELINE
# Complete implementation with real LLM integration for disaster management
# ============================================================================

import json
import os
import numpy as np
from datetime import datetime, timedelta
import random
import math
import pandas as pd
from collections import defaultdict
import requests

# ============================================================================
# 1. ENVIRONMENTAL DATA SIMULATOR
# ============================================================================

class EnvironmentalDataSimulator:
    """Simulates comprehensive environmental data for flood prediction"""
    
    def __init__(self):
        self.regions = [
            "Downtown Area", "Riverside District", "Industrial Zone", 
            "Residential Suburbs", "Agricultural Valley", "Coastal Region"
        ]
        
        # Topographical data for each region
        self.topography = {
            "Downtown Area": {"elevation": 15, "population_density": 8500, "infrastructure": "high"},
            "Riverside District": {"elevation": 8, "population_density": 3200, "infrastructure": "medium"},
            "Industrial Zone": {"elevation": 12, "population_density": 1200, "infrastructure": "high"},
            "Residential Suburbs": {"elevation": 22, "population_density": 2800, "infrastructure": "medium"},
            "Agricultural Valley": {"elevation": 5, "population_density": 350, "infrastructure": "low"},
            "Coastal Region": {"elevation": 3, "population_density": 1800, "infrastructure": "medium"}
        }
        
        print("🌍 Environmental Data Simulator initialized")
        print(f"   Monitoring {len(self.regions)} regions")
    
    def generate_environmental_data(self, flood_scenario="normal"):
        """
        Generate comprehensive environmental data for all regions
        
        Args:
            flood_scenario: "normal", "moderate_risk", "high_risk", "extreme_risk"
        """
        
        environmental_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": flood_scenario,
            "regions": {}
        }
        
        # Generate data for each region
        for region in self.regions:
            region_data = self._generate_region_data(region, flood_scenario)
            environmental_data["regions"][region] = region_data
        
        # Generate forecast data
        environmental_data["forecast"] = self._generate_forecast_data(flood_scenario)
        
        return environmental_data
    
    def _generate_region_data(self, region, scenario):
        """Generate data for a specific region"""
        
        topo = self.topography[region]
        
        # Base conditions
        base_rainfall = 2.0  # mm/hour
        base_water_level = 1.0  # meters
        base_humidity = 65  # %
        base_temperature = 20  # °C
        
        # Scenario-based modifications
        if scenario == "moderate_risk":
            rainfall_multiplier = random.uniform(2.0, 3.5)
            water_level_increase = random.uniform(0.8, 1.5)
            humidity_increase = random.uniform(10, 20)
        elif scenario == "high_risk":
            rainfall_multiplier = random.uniform(3.5, 6.0)
            water_level_increase = random.uniform(1.5, 2.8)
            humidity_increase = random.uniform(20, 30)
        elif scenario == "extreme_risk":
            rainfall_multiplier = random.uniform(6.0, 12.0)
            water_level_increase = random.uniform(2.8, 5.0)
            humidity_increase = random.uniform(30, 40)
        else:  # normal
            rainfall_multiplier = random.uniform(0.5, 1.2)
            water_level_increase = random.uniform(-0.2, 0.3)
            humidity_increase = random.uniform(-5, 5)
        
        # Elevation-based adjustments (lower areas more vulnerable)
        elevation_factor = max(0.3, 1.0 - (topo["elevation"] / 50))
        
        # Generate weather data
        weather_data = {
            "rainfall_current": round(base_rainfall * rainfall_multiplier, 2),
            "rainfall_24h": round(base_rainfall * rainfall_multiplier * 24, 1),
            "humidity": round(base_humidity + humidity_increase, 1),
            "temperature": round(base_temperature + random.uniform(-3, 3), 1),
            "barometric_pressure": round(1013 + random.uniform(-15, 15), 1)
        }
        
        # Generate seismic data
        seismic_data = {
            "recent_earthquake_magnitude": round(random.uniform(0, 4.5), 1),
            "earthquake_depth": round(random.uniform(5, 50), 1),
            "seismic_activity_level": random.choice(["minimal", "low", "moderate"]),
            "days_since_last_earthquake": random.randint(1, 90)
        }
        
        # Generate wind data
        wind_data = {
            "surface_wind_speed": round(random.uniform(5, 45), 1),
            "surface_wind_direction": random.randint(0, 360),
            "sea_wind_speed": round(random.uniform(8, 35), 1),
            "sea_wind_direction": random.randint(0, 360),
            "wind_gust_speed": round(random.uniform(15, 60), 1)
        }
        
        # Generate water level data
        water_level_data = {
            "current_level": round(base_water_level + water_level_increase * elevation_factor, 2),
            "24h_change": round(water_level_increase * elevation_factor, 2),
            "historical_average": round(base_water_level, 2),
            "flood_stage": round(base_water_level + 2.0, 2),
            "critical_stage": round(base_water_level + 4.0, 2)
        }
        
        # Generate rainfall data (local and upstream)
        rainfall_data = {
            "local_rainfall": weather_data["rainfall_current"],
            "upstream_rainfall": round(weather_data["rainfall_current"] * random.uniform(0.8, 1.5), 2),
            "local_24h_total": weather_data["rainfall_24h"],
            "upstream_24h_total": round(weather_data["rainfall_24h"] * random.uniform(0.8, 1.5), 1)
        }
        
        return {
            "region": region,
            "topography": topo,
            "weather": weather_data,
            "seismic": seismic_data,
            "wind": wind_data,
            "water_level": water_level_data,
            "rainfall": rainfall_data
        }
    
    def _generate_forecast_data(self, scenario):
        """Generate forecast data for next 72 hours"""
        
        forecast_data = []
        
        for hour in range(1, 73):  # 72-hour forecast
            # Base forecast values
            base_rainfall = 1.5
            
            # Scenario adjustments
            if scenario == "extreme_risk":
                rainfall_forecast = base_rainfall * random.uniform(4.0, 10.0)
            elif scenario == "high_risk":
                rainfall_forecast = base_rainfall * random.uniform(2.5, 5.0)
            elif scenario == "moderate_risk":
                rainfall_forecast = base_rainfall * random.uniform(1.5, 3.0)
            else:
                rainfall_forecast = base_rainfall * random.uniform(0.2, 1.2)
            
            # Add some temporal variation
            time_factor = 1.0 + 0.3 * math.sin(hour * 0.1)
            rainfall_forecast *= time_factor
            
            forecast_point = {
                "hour": hour,
                "rainfall_forecast": round(max(0, rainfall_forecast), 2),
                "confidence": round(random.uniform(0.7, 0.95), 2)
            }
            
            forecast_data.append(forecast_point)
        
        return forecast_data

# ============================================================================
# 2. FLOOD PREDICTION ENGINE
# ============================================================================

class FloodPredictionEngine:
    """Advanced flood prediction using environmental data aggregation"""
    
    def __init__(self):
        self.risk_thresholds = {
            "rainfall": [10, 25, 50],  # mm/hour
            "water_level": [2.0, 3.5, 5.0],  # meters above normal
            "rainfall_24h": [50, 100, 200],  # mm/24h
            "upstream_factor": [1.2, 1.5, 2.0],  # upstream multiplier
            "wind_speed": [25, 40, 60]  # km/h
        }
        
        print("🔮 Flood Prediction Engine initialized")
    
    def predict_flood_severity(self, environmental_data):
        """
        Implement Algorithm Steps 1-3: Feature aggregation, prediction, risk interpretation
        """
        
        print("📊 Analyzing environmental data for flood prediction...")
        
        # Step 1: Feature Aggregation
        aggregated_features = self._aggregate_features(environmental_data)
        
        # Step 2: Flood Severity Prediction
        severity_scores = {}
        overall_severity = 0
        
        for region, features in aggregated_features.items():
            region_severity = self._predict_region_severity(features)
            severity_scores[region] = region_severity
            
            # Weight by population density for overall severity
            population_weight = features["population_density"] / 10000
            overall_severity += region_severity * population_weight
        
        # Normalize overall severity
        total_weight = sum(f["population_density"] / 10000 for f in aggregated_features.values())
        overall_severity = min(overall_severity / total_weight, 1.0)
        
        # Step 3: Risk Interpretation
        risk_level = self._interpret_risk_level(overall_severity)
        
        prediction_results = {
            "overall_severity": round(overall_severity, 3),
            "risk_level": risk_level,
            "region_severities": severity_scores,
            "aggregated_features": aggregated_features,
            "high_risk_regions": [region for region, score in severity_scores.items() if score > 0.6]
        }
        
        return prediction_results
    
    def _aggregate_features(self, environmental_data):
        """Aggregate input data into unified feature vectors"""
        
        aggregated = {}
        
        for region, data in environmental_data["regions"].items():
            # Extract and normalize features
            features = {
                "region": region,
                "elevation": data["topography"]["elevation"],
                "population_density": data["topography"]["population_density"],
                "infrastructure": data["topography"]["infrastructure"],
                
                # Weather features
                "rainfall_current": data["weather"]["rainfall_current"],
                "rainfall_24h": data["weather"]["rainfall_24h"],
                "humidity": data["weather"]["humidity"],
                "temperature": data["weather"]["temperature"],
                "barometric_pressure": data["weather"]["barometric_pressure"],
                
                # Seismic features
                "earthquake_magnitude": data["seismic"]["recent_earthquake_magnitude"],
                "earthquake_depth": data["seismic"]["earthquake_depth"],
                
                # Wind features
                "surface_wind_speed": data["wind"]["surface_wind_speed"],
                "sea_wind_speed": data["wind"]["sea_wind_speed"],
                "wind_gust_speed": data["wind"]["wind_gust_speed"],
                
                # Water level features
                "current_water_level": data["water_level"]["current_level"],
                "water_level_change": data["water_level"]["24h_change"],
                "flood_stage_ratio": data["water_level"]["current_level"] / data["water_level"]["flood_stage"],
                
                # Rainfall features
                "local_rainfall": data["rainfall"]["local_rainfall"],
                "upstream_rainfall": data["rainfall"]["upstream_rainfall"],
                "upstream_factor": data["rainfall"]["upstream_rainfall"] / max(data["rainfall"]["local_rainfall"], 0.1)
            }
            
            aggregated[region] = features
        
        return aggregated
    
    def _predict_region_severity(self, features):
        """Predict flood severity for a specific region"""
        
        # Weighted feature scoring
        severity_factors = []
        
        # Rainfall factor (40% weight)
        rainfall_score = min(features["rainfall_current"] / 50, 1.0)
        severity_factors.append(0.4 * rainfall_score)
        
        # Water level factor (30% weight)
        water_level_score = min(features["water_level_change"] / 5.0, 1.0)
        severity_factors.append(0.3 * water_level_score)
        
        # Upstream factor (15% weight)
        upstream_score = min((features["upstream_factor"] - 1.0) / 2.0, 1.0)
        severity_factors.append(0.15 * max(0, upstream_score))
        
        # Wind factor (10% weight)
        wind_score = min(features["surface_wind_speed"] / 60, 1.0)
        severity_factors.append(0.1 * wind_score)
        
        # Elevation adjustment (vulnerability factor)
        elevation_vulnerability = max(0.3, 1.0 - (features["elevation"] / 50))
        
        # Calculate base severity
        base_severity = sum(severity_factors) * elevation_vulnerability
        
        # Infrastructure adjustment
        infrastructure_factor = {
            "high": 0.8,    # Better infrastructure = lower risk
            "medium": 1.0,
            "low": 1.2      # Poor infrastructure = higher risk
        }.get(features["infrastructure"], 1.0)
        
        final_severity = min(base_severity * infrastructure_factor, 1.0)
        
        return round(final_severity, 3)
    
    def _interpret_risk_level(self, severity):
        """Map severity score to qualitative risk level"""
        
        if severity >= 0.8:
            return "Critical"
        elif severity >= 0.6:
            return "Severe"
        elif severity >= 0.3:
            return "Moderate"
        else:
            return "Low"

# ============================================================================
# 3. REAL LLM INTEGRATION FOR DISASTER RESPONSE
# ============================================================================

class DisasterResponseLLM:
    """Real LLM integration for flood disaster response planning"""
    
    def __init__(self, llm_type="rule_based"):
        self.llm_type = llm_type
        self.model_available = False
        
        if llm_type == "transformers":
            self.setup_transformers()
        elif llm_type == "ollama":
            self.setup_ollama()
        else:
            # Default to expert disaster management system
            self.model_available = True
            print("🚨 Using expert disaster management system")
    
    def setup_transformers(self):
        """Setup Hugging Face Transformers"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Use a free model
            model_name = "microsoft/DialoGPT-medium"
            print(f"📥 Loading {model_name} for disaster response...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_available = True
            print("✅ Transformers model loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Transformers error: {e}")
            print("   Using expert disaster management system")
    
    def setup_ollama(self):
        """Setup Ollama LLM"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                
                preferred_models = ['llama2', 'mistral', 'phi']
                self.selected_model = None
                
                for model in preferred_models:
                    if any(model in m for m in available_models):
                        self.selected_model = model
                        break
                
                if self.selected_model:
                    self.model_available = True
                    print(f"✅ Using Ollama model: {self.selected_model}")
                else:
                    print("⚠️  No suitable Ollama models found")
        except:
            print("❌ Ollama not available")
    
    def create_disaster_response_prompt(self, prediction_results, environmental_data):
        """
        Algorithm Step 4: Construct structured prompt for disaster response
        """
        
        severity = prediction_results["overall_severity"]
        risk_level = prediction_results["risk_level"]
        high_risk_regions = prediction_results["high_risk_regions"]
        
        # Format regional details
        regional_details = []
        for region, score in prediction_results["region_severities"].items():
            region_data = environmental_data["regions"][region]
            details = f"- {region}: Severity {score:.3f}, Population {region_data['topography']['population_density']}, Elevation {region_data['topography']['elevation']}m"
            regional_details.append(details)
        
        regional_text = "\n".join(regional_details)
        
        # Format high-risk regions
        high_risk_text = ", ".join(high_risk_regions) if high_risk_regions else "None"
        
        # Format forecast summary
        forecast_data = environmental_data["forecast"]
        next_24h_rainfall = sum(f["rainfall_forecast"] for f in forecast_data[:24])
        
        prompt = f"""You are an expert disaster management coordinator with 20 years of experience in flood emergency response and evacuation planning.

FLOOD RISK ASSESSMENT SUMMARY:
Analysis Date: {environmental_data['timestamp']}
Overall Flood Severity: {severity:.3f} (scale 0-1)
Risk Level: {risk_level.upper()}
Scenario: {environmental_data['scenario']}

REGIONAL ANALYSIS:
{regional_text}

HIGH-RISK REGIONS: {high_risk_text}

ENVIRONMENTAL CONDITIONS:
Weather Forecast:
- Next 24h rainfall forecast: {next_24h_rainfall:.1f} mm
- Current conditions: {environmental_data['scenario']} scenario

Critical Factors:
- Multiple regions under assessment
- Population density considerations included
- Topographical vulnerability factors analyzed
- Upstream and local rainfall patterns evaluated

DISASTER RESPONSE REQUIREMENTS:
Given the {risk_level.upper()} risk level (severity {severity:.3f}), provide comprehensive disaster response planning:

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

        return prompt
    
    def generate_disaster_response_plan(self, prompt):
        """
        Algorithm Step 5: Generate comprehensive disaster response plan
        """
        
        print("🚨 Generating disaster response plan...")
        
        if self.llm_type == "transformers" and self.model_available:
            return self._transformers_generate(prompt)
        elif self.llm_type == "ollama" and self.model_available:
            return self._ollama_generate(prompt)
        else:
            return self._expert_disaster_response(prompt)
    
    def _transformers_generate(self, prompt):
        """Generate using Transformers"""
        try:
            # Simplify prompt for model
            severity = self._extract_severity(prompt)
            risk_level = self._extract_risk_level(prompt)
            
            simplified_prompt = f"Flood emergency response plan: {risk_level} risk level, severity {severity:.3f}. Evacuation and emergency actions:"
            
            inputs = self.tokenizer.encode(simplified_prompt, return_tensors="pt")
            
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + 400,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(simplified_prompt):].strip()
            
            # If response is too short, fall back to expert system
            if len(response) < 100:
                return self._expert_disaster_response(prompt)
            
            return response
            
        except Exception as e:
            print(f"❌ Transformers generation error: {e}")
            return self._expert_disaster_response(prompt)
    
    def _ollama_generate(self, prompt):
        """Generate using Ollama"""
        try:
            payload = {
                "model": self.selected_model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return self._expert_disaster_response(prompt)
                
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return self._expert_disaster_response(prompt)
    
    def _extract_severity(self, prompt):
        """Extract severity from prompt"""
        try:
            line = [l for l in prompt.split('\n') if 'Overall Flood Severity:' in l][0]
            return float(line.split(':')[1].split()[0])
        except:
            return 0.5
    
    def _extract_risk_level(self, prompt):
        """Extract risk level from prompt"""
        try:
            line = [l for l in prompt.split('\n') if 'Risk Level:' in l][0]
            return line.split(':')[1].strip()
        except:
            return "MODERATE"
    
    def _expert_disaster_response(self, prompt):
        """Expert disaster management system"""
        
        print("👨‍💼 Using expert disaster management analysis")
        
        # Extract key parameters
        severity = self._extract_severity(prompt)
        risk_level = self._extract_risk_level(prompt)
        
        # Extract high-risk regions
        high_risk_regions = []
        lines = prompt.split('\n')
        for line in lines:
            if 'HIGH-RISK REGIONS:' in line:
                regions_text = line.split(':')[1].strip()
                if regions_text != "None":
                    high_risk_regions = [r.strip() for r in regions_text.split(',')]
        
        # Generate comprehensive disaster response plan
        return self._generate_expert_response_plan(severity, risk_level, high_risk_regions)
    
    def _generate_expert_response_plan(self, severity, risk_level, high_risk_regions):
        """Generate expert-level disaster response plan"""
        
        # Determine response parameters
        if severity >= 0.8:
            alert_level = "RED - EXTREME EMERGENCY"
            response_time = "IMMEDIATE - 0-2 hours"
            evacuation_scope = "MANDATORY MASS EVACUATION"
        elif severity >= 0.6:
            alert_level = "ORANGE - SEVERE WARNING"
            response_time = "URGENT - 2-6 hours"
            evacuation_scope = "VOLUNTARY EVACUATION RECOMMENDED"
        elif severity >= 0.3:
            alert_level = "YELLOW - MODERATE WARNING"
            response_time = "PROMPT - 6-12 hours"
            evacuation_scope = "PRECAUTIONARY MEASURES"
        else:
            alert_level = "GREEN - ADVISORY"
            response_time = "ROUTINE - 12-24 hours"
            evacuation_scope = "MONITORING PHASE"
        
        # Generate detailed response plan
        response_plan = f"""
COMPREHENSIVE FLOOD DISASTER RESPONSE PLAN

EXECUTIVE SUMMARY:
Alert Level: {alert_level}
Flood Severity: {severity:.3f}/1.0
Risk Classification: {risk_level.upper()}
Response Timeline: {response_time}
Evacuation Status: {evacuation_scope}

1. IMMEDIATE ACTIONS ({response_time}):

Emergency Operations:
• Activate Emergency Operations Center (EOC) at {alert_level} level
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
{self._format_evacuation_priorities(high_risk_regions, severity)}

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

Public Alert Level: {alert_level}

Emergency Broadcast Message:
"FLOOD WARNING - {alert_level}
This is an official emergency alert. {"IMMEDIATE EVACUATION REQUIRED" if severity >= 0.8 else "FLOOD WARNING IN EFFECT"} for the following areas: {', '.join(high_risk_regions) if high_risk_regions else 'multiple regions'}.
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
{self._format_priority_areas(high_risk_regions, severity)}

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
This {risk_level.upper()} flood risk scenario (severity {severity:.3f}) requires {response_time.lower()} coordinated emergency response. The plan prioritizes life safety through {"immediate evacuation" if severity >= 0.8 else "protective actions"}, resource mobilization, and multi-agency coordination. Success depends on rapid implementation of evacuation procedures and effective public communication.

NEXT STEPS:
1. {"Execute immediate evacuation procedures" if severity >= 0.8 else "Implement monitoring and preparation protocols"}
2. {"Activate all emergency response agencies" if severity >= 0.8 else "Alert emergency response agencies"}
3. {"Begin mass public notifications" if severity >= 0.8 else "Issue public advisories"}
4. {"Establish emergency operations at full capacity" if severity >= 0.8 else "Prepare emergency operations"}
5. {"Monitor flood conditions continuously" if severity >= 0.6 else "Continue environmental monitoring"}

This disaster response plan follows established emergency management protocols and is designed to protect lives and minimize property damage during flood events.
"""
        
        return response_plan.strip()
    
    def _format_evacuation_priorities(self, high_risk_regions, severity):
        """Format evacuation priorities based on risk regions"""
        
        if not high_risk_regions:
            return "• No specific high-risk regions identified - monitor all areas"
        
        priorities = []
        for i, region in enumerate(high_risk_regions, 1):
            if severity >= 0.8:
                priorities.append(f"• PRIORITY {i}: {region} - IMMEDIATE MANDATORY EVACUATION")
            elif severity >= 0.6:
                priorities.append(f"• PRIORITY {i}: {region} - VOLUNTARY EVACUATION RECOMMENDED")
            else:
                priorities.append(f"• PRIORITY {i}: {region} - ENHANCED MONITORING")
        
        return "\n".join(priorities)
    
    def _format_priority_areas(self, high_risk_regions, severity):
        """Format priority areas for response"""
        
        if not high_risk_regions:
            return "• All regions under general monitoring protocol"
        
        priorities = []
        for i, region in enumerate(high_risk_regions, 1):
            urgency = "CRITICAL" if severity >= 0.8 else "HIGH" if severity >= 0.6 else "MODERATE"
            priorities.append(f"• {region} ({urgency} priority)")
        
        return "\n".join(priorities)

# ============================================================================
# 4. COMPLETE FLOOD FORECASTING PIPELINE
# ============================================================================

class FloodForecastingPipeline:
    """Complete flood forecasting and disaster response pipeline"""
    
    def __init__(self, llm_type="rule_based"):
        self.data_simulator = EnvironmentalDataSimulator()
        self.prediction_engine = FloodPredictionEngine()
        self.response_llm = DisasterResponseLLM(llm_type)
        
        # Create output directories
        os.makedirs("flood_data", exist_ok=True)
        os.makedirs("disaster_reports", exist_ok=True)
        
        print(f"🌊 Flood Forecasting Pipeline initialized with {llm_type} LLM")
    
    def run_flood_analysis(self, flood_scenario="moderate_risk"):
        """
        Run complete flood forecasting and disaster response planning
        Implements all algorithm steps
        """
        
        print("🌊 FLOOD FORECASTING AND EARLY ACTION PLANNING")
        print("=" * 70)
        
        # Generate environmental data
        print(f"\n1. Generating environmental data (scenario: {flood_scenario})...")
        environmental_data = self.data_simulator.generate_environmental_data(flood_scenario)
        
        region_count = len(environmental_data["regions"])
        print(f"   🌍 Monitoring {region_count} regions")
        print(f"   📊 72-hour forecast generated")
        
        # Algorithm Steps 1-3: Predict flood severity
        print("\n2. Predicting flood severity...")
        prediction_results = self.prediction_engine.predict_flood_severity(environmental_data)
        
        severity = prediction_results["overall_severity"]
        risk_level = prediction_results["risk_level"]
        
        print(f"   🔮 Overall severity: {severity:.3f}")
        print(f"   🚨 Risk level: {risk_level}")
        print(f"   ⚠️  High-risk regions: {len(prediction_results['high_risk_regions'])}")
        
        # Algorithm Step 4: Create disaster response prompt
        print("\n3. Constructing disaster response prompt...")
        prompt = self.response_llm.create_disaster_response_prompt(
            prediction_results, environmental_data
        )
        
        # Algorithm Step 5: Generate disaster response plan
        print("\n4. Generating disaster response plan...")
        response_plan = self.response_llm.generate_disaster_response_plan(prompt)
        
        # Compile final results
        final_results = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": flood_scenario,
            "llm_type": self.response_llm.llm_type,
            "environmental_data": environmental_data,
            "prediction_results": prediction_results,
            "disaster_response_plan": response_plan,
            "algorithm_outputs": {
                "lambda": severity,
                "r": risk_level,
                "A": response_plan
            }
        }
        
        # Display and save results
        self._display_results(final_results)
        self._save_results(final_results)
        
        return final_results
    
    def _display_results(self, results):
        """Display flood analysis results"""
        
        print("\n" + "=" * 70)
        print("🌊 FLOOD FORECASTING ANALYSIS RESULTS")
        print("=" * 70)
        
        # Basic information
        print(f"📅 Analysis Date: {results['analysis_date']}")
        print(f"🎭 Scenario: {results['scenario']}")
        print(f"🤖 LLM Type: {results['llm_type']}")
        
        # Prediction results
        prediction = results['prediction_results']
        print(f"\n🔮 FLOOD PREDICTION SUMMARY:")
        print(f"   Overall Severity (λ): {prediction['overall_severity']:.3f}")
        print(f"   Risk Level (r): {prediction['risk_level']}")
        print(f"   High-risk regions: {len(prediction['high_risk_regions'])}")
        
        # Regional breakdown
        print(f"\n🌍 REGIONAL SEVERITY BREAKDOWN:")
        for region, score in prediction['region_severities'].items():
            status = "⚠️ HIGH RISK" if score > 0.6 else "📊 MODERATE" if score > 0.3 else "✅ LOW RISK"
            print(f"   {region}: {score:.3f} {status}")
        
        # High-risk regions
        if prediction['high_risk_regions']:
            print(f"\n🚨 HIGH-RISK REGIONS REQUIRING IMMEDIATE ATTENTION:")
            for region in prediction['high_risk_regions']:
                print(f"   • {region}")
        
        # Disaster response plan preview
        print(f"\n🚨 DISASTER RESPONSE PLAN PREVIEW:")
        print("-" * 60)
        response_lines = results['disaster_response_plan'].split('\n')
        for line in response_lines[:15]:
            if line.strip():
                print(f"   {line.strip()}")
        
        if len(response_lines) > 15:
            print("   ... (continued in saved report)")
        print("-" * 60)
    
    def _save_results(self, results):
        """Save flood analysis results"""
        
        # Save detailed JSON report
        timestamp = results['analysis_date'].replace(' ', '_').replace(':', '-')
        json_filename = f"disaster_reports/flood_analysis_{timestamp}.json"
        
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save disaster response plan
        response_filename = f"disaster_reports/response_plan_{timestamp}.txt"
        with open(response_filename, 'w') as f:
            f.write("FLOOD DISASTER RESPONSE PLAN\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis Date: {results['analysis_date']}\n")
            f.write(f"Scenario: {results['scenario']}\n")
            f.write(f"LLM Model: {results['llm_type']}\n\n")
            
            f.write("ALGORITHM OUTPUTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Flood Severity (λ): {results['algorithm_outputs']['lambda']:.3f}\n")
            f.write(f"Risk Level (r): {results['algorithm_outputs']['r']}\n\n")
            
            f.write("REGIONAL ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            for region, score in results['prediction_results']['region_severities'].items():
                f.write(f"{region}: {score:.3f}\n")
            
            f.write(f"\nDISASTER RESPONSE PLAN:\n")
            f.write("-" * 30 + "\n")
            f.write(results['disaster_response_plan'])
        
        # Save environmental data to CSV
        csv_filename = f"flood_data/environmental_data_{timestamp}.csv"
        env_data = []
        for region, data in results['environmental_data']['regions'].items():
            row = {
                'region': region,
                'elevation': data['topography']['elevation'],
                'population_density': data['topography']['population_density'],
                'rainfall_current': data['weather']['rainfall_current'],
                'rainfall_24h': data['weather']['rainfall_24h'],
                'water_level': data['water_level']['current_level'],
                'water_level_change': data['water_level']['24h_change'],
                'wind_speed': data['wind']['surface_wind_speed'],
                'severity_score': results['prediction_results']['region_severities'][region]
            }
            env_data.append(row)
        
        df = pd.DataFrame(env_data)
        df.to_csv(csv_filename, index=False)
        
        print(f"\n✅ Reports saved:")
        print(f"   📄 Detailed analysis: {json_filename}")
        print(f"   📋 Response plan: {response_filename}")
        print(f"   📊 Environmental data: {csv_filename}")

# ============================================================================
# 5. MAIN EXECUTION WITH SCENARIO SELECTION
# ============================================================================

def print_flood_scenarios():
    """Print available flood scenarios"""
    
    print("🌊 AVAILABLE FLOOD SCENARIOS:")
    print("=" * 50)
    
    scenarios = {
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
    
    for i, (scenario, details) in enumerate(scenarios.items(), 1):
        print(f"\n{i}. {scenario.upper()}")
        print(f"   Description: {details['description']}")
        print(f"   Characteristics: {details['characteristics']}")
        print(f"   Expected Severity: {details['expected_severity']}")
        print(f"   Risk Level: {details['risk_level']}")

def print_llm_options():
    """Print LLM options for disaster response"""
    
    print("\n🤖 LLM OPTIONS FOR DISASTER RESPONSE:")
    print("=" * 50)
    
    print("\n1. EXPERT DISASTER MANAGEMENT SYSTEM (Recommended)")
    print("   ✅ Professional emergency management protocols")
    print("   ✅ Comprehensive evacuation planning")
    print("   ✅ Multi-agency coordination procedures")
    print("   ✅ Real-time response capabilities")
    
    print("\n2. HUGGING FACE TRANSFORMERS")
    print("   📥 Requires: pip install transformers torch")
    print("   ✅ AI-powered response generation")
    print("   ✅ Free models available")
    
    print("\n3. OLLAMA (Local LLM)")
    print("   📥 Requires: Ollama installation")
    print("   ✅ High-quality disaster response")
    print("   ✅ Privacy-focused processing")

def main():
    """Main execution function"""
    
    print("🌊 FLOOD FORECASTING AND EARLY ACTION PLANNING PIPELINE")
    print("AI-Powered Disaster Management and Emergency Response System")
    print("=" * 70)
    
    # Show scenario options
    print_flood_scenarios()
    
    # Show LLM options
    print_llm_options()
    
    # Get user selections
    print("\n" + "=" * 70)
    print("SYSTEM CONFIGURATION:")
    
    # Select scenario
    print("\nSelect flood scenario:")
    print("1. Normal conditions")
    print("2. Moderate risk")
    print("3. High risk")
    print("4. Extreme risk")
    
    scenario_choice = input("\nEnter scenario choice (1-4) [2]: ").strip()
    scenario_mapping = {
        "1": "normal",
        "2": "moderate_risk",
        "3": "high_risk",
        "4": "extreme_risk",
        "": "moderate_risk"  # Default
    }
    
    selected_scenario = scenario_mapping.get(scenario_choice, "moderate_risk")
    
    # Select LLM type
    print("\nSelect LLM type:")
    print("1. Expert Disaster Management System")
    print("2. Hugging Face Transformers")
    print("3. Ollama")
    
    llm_choice = input("\nEnter LLM choice (1-3) [1]: ").strip()
    llm_mapping = {
        "1": "rule_based",
        "2": "transformers",
        "3": "ollama",
        "": "rule_based"  # Default
    }
    
    selected_llm = llm_mapping.get(llm_choice, "rule_based")
    
    print(f"\n🌊 Selected scenario: {selected_scenario}")
    print(f"🤖 Selected LLM: {selected_llm}")
    
    # Initialize pipeline
    print(f"\n🔧 Initializing flood forecasting pipeline...")
    pipeline = FloodForecastingPipeline(selected_llm)
    
    # Run analysis
    try:
        print(f"\n🔍 Running flood analysis...")
        
        results = pipeline.run_flood_analysis(selected_scenario)
        
        print("\n" + "=" * 70)
        print("🎉 FLOOD FORECASTING ANALYSIS COMPLETE!")
        print("=" * 70)
        
        # Display key findings
        print("🔬 KEY FINDINGS:")
        print(f"   Flood Severity (λ): {results['algorithm_outputs']['lambda']:.3f}")
        print(f"   Risk Level (r): {results['algorithm_outputs']['r']}")
        print(f"   High-risk regions: {len(results['prediction_results']['high_risk_regions'])}")
        
        # Alert level summary
        risk_level = results['algorithm_outputs']['r']
        if risk_level == "Critical":
            print(f"\n🚨 CRITICAL ALERT: Immediate evacuation and emergency response required!")
        elif risk_level == "Severe":
            print(f"\n⚠️  SEVERE WARNING: Urgent flood preparedness and possible evacuation needed")
        elif risk_level == "Moderate":
            print(f"\n📋 MODERATE WARNING: Enhanced monitoring and preparedness recommended")
        else:
            print(f"\n✅ LOW RISK: Continue routine monitoring")
        
        print("\n📁 REPORTS GENERATED:")
        print("   📄 Detailed flood analysis")
        print("   📋 Disaster response plan")
        print("   📊 Environmental data export")
        print("   🚨 Emergency coordination protocols")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("Please check your configuration and try again.")
        return None

# ============================================================================
# 6. COMPREHENSIVE SCENARIO TESTING
# ============================================================================

def demo_all_flood_scenarios():
    """Demonstrate all flood scenarios"""
    
    print("🧪 DEMONSTRATING ALL FLOOD SCENARIOS")
    print("=" * 60)
    
    scenarios = ["normal", "moderate_risk", "high_risk", "extreme_risk"]
    pipeline = FloodForecastingPipeline("rule_based")
    
    results_summary = []
    
    for scenario in scenarios:
        print(f"\n🌊 Testing scenario: {scenario}")
        print("-" * 40)
        
        results = pipeline.run_flood_analysis(scenario)
        
        results_summary.append({
            "scenario": scenario,
            "severity": results['algorithm_outputs']['lambda'],
            "risk_level": results['algorithm_outputs']['r'],
            "high_risk_regions": len(results['prediction_results']['high_risk_regions'])
        })
        
        print(f"   λ: {results['algorithm_outputs']['lambda']:.3f}")
        print(f"   Risk: {results['algorithm_outputs']['r']}")
        print(f"   High-risk regions: {len(results['prediction_results']['high_risk_regions'])}")
    
    # Display summary
    print("\n FLOOD SCENARIO COMPARISON:")
    print("=" * 60)
    print("Scenario        | Severity | Risk Level | High-Risk Regions")
    print("-" * 60)
    for result in results_summary:
        print(f"{result['scenario']:15} | {result['severity']:.3f}    | {result['risk_level']:10} | {result['high_risk_regions']}")
    
    return results_summary

if __name__ == "__main__":
    # Run the complete flood forecasting pipeline
    results = main()
    
    # Optionally run demo of all scenarios
    if results:
        demo_choice = input("\nRun comprehensive scenario testing? (y/n) [n]: ").strip().lower()
        if demo_choice == 'y':
            demo_all_flood_scenarios()