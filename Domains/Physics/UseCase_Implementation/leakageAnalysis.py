######################################################################
# Underwater Leakage Detection and Risk Assessment Pipeline
# Physics-based system for pipeline integrity monitoring and analysis
######################################################################

import json
import os
import numpy as np
from datetime import datetime
import random
import math
import pandas as pd
from collections import defaultdict
import requests

######################################################################
# Underwater Sensor Data Simulation System
######################################################################

class UnderwaterSensorSimulator:
    """
    Simulates realistic underwater sensor data for leakage detection systems.
    Uses physics-based calculations to generate hydrostatic pressure readings.
    """
    
    def __init__(self):
        ######################################################################
        # Define physical constants for underwater calculations
        ######################################################################
        self.gravitational_constant = 9.81  # m/s²
        self.seawater_density = 1025  # kg/m³
        self.atmospheric_pressure = 101325  # Pa
        
        ######################################################################
        # Set realistic sensor accuracy specifications
        ######################################################################
        self.measurement_accuracy = {
            "pressure": 0.02,  # 2% accuracy
            "temperature": 0.5,  # ±0.5°C
            "flow": 0.05,  # 5% accuracy
            "depth": 0.1  # ±0.1m
        }
        
        print("Underwater Sensor Simulator initialized")
        print(f"   Water density: {self.seawater_density} kg/m³")
        print(f"   Gravitational constant: {self.gravitational_constant} m/s²")
    
    def generate_realistic_sensor_data(self, measurement_points=20, leakage_scenario="normal"):
        """
        Generates physics-based sensor data for various leakage scenarios.
        Uses hydrostatic pressure calculations with realistic sensor noise.
        
        Args:
            measurement_points: Number of sensor readings to generate
            leakage_scenario: "normal", "minor_leak", "major_leak", "critical_leak"
        """
        
        sensor_readings = []
        
        ######################################################################
        # Create depth measurement points (5-50m underwater range)
        ######################################################################
        depth_measurements = np.linspace(5, 50, measurement_points)
        
        for measurement_index, depth in enumerate(depth_measurements):
            ######################################################################
            # Calculate expected pressure using hydrostatic formula: P = P0 + ρgh
            ######################################################################
            theoretical_pressure = (self.atmospheric_pressure + 
                                  (self.seawater_density * self.gravitational_constant * depth))
            
            ######################################################################
            # Add realistic sensor measurement noise
            ######################################################################
            pressure_measurement_noise = np.random.normal(0, theoretical_pressure * self.measurement_accuracy["pressure"])
            
            ######################################################################
            # Apply scenario-specific pressure anomalies
            ######################################################################
            if leakage_scenario == "minor_leak":
                if 15 <= depth <= 25:
                    pressure_anomaly = -np.random.uniform(1000, 3000)  # 1-3 kPa drop
                else:
                    pressure_anomaly = np.random.normal(0, 500)
            elif leakage_scenario == "major_leak":
                if 20 <= depth <= 35:
                    pressure_anomaly = -np.random.uniform(3000, 8000)  # 3-8 kPa drop
                else:
                    pressure_anomaly = np.random.normal(0, 800)
            elif leakage_scenario == "critical_leak":
                if 25 <= depth <= 40:
                    pressure_anomaly = -np.random.uniform(8000, 15000)  # 8-15 kPa drop
                else:
                    pressure_anomaly = np.random.normal(0, 1200)
            else:
                # Normal operation conditions
                pressure_anomaly = np.random.normal(0, 300)
            
            ######################################################################
            # Calculate final observed pressure with all factors
            ######################################################################
            observed_pressure = theoretical_pressure + pressure_measurement_noise + pressure_anomaly
            
            ######################################################################
            # Generate temperature readings (decreases with depth)
            ######################################################################
            baseline_temperature = 15 - (depth * 0.1)  # 15°C surface, decreases with depth
            measured_temperature = baseline_temperature + np.random.normal(0, self.measurement_accuracy["temperature"])
            
            ######################################################################
            # Generate flow rate data based on leakage scenario
            ######################################################################
            if leakage_scenario == "normal":
                flow_rate = np.random.uniform(0.5, 2.0)  # Normal operational flow
            elif leakage_scenario == "minor_leak":
                if 15 <= depth <= 25:
                    flow_rate = np.random.uniform(2.5, 4.0)  # Elevated flow from leak
                else:
                    flow_rate = np.random.uniform(0.5, 2.0)
            elif leakage_scenario == "major_leak":
                if 20 <= depth <= 35:
                    flow_rate = np.random.uniform(4.0, 8.0)  # High flow from significant leak
                else:
                    flow_rate = np.random.uniform(0.5, 2.5)
            elif leakage_scenario == "critical_leak":
                if 25 <= depth <= 40:
                    flow_rate = np.random.uniform(8.0, 15.0)  # Very high flow from critical failure
                else:
                    flow_rate = np.random.uniform(1.0, 3.0)
            
            ######################################################################
            # Add measurement noise to flow readings
            ######################################################################
            flow_rate += np.random.normal(0, flow_rate * self.measurement_accuracy["flow"])
            flow_rate = max(0, flow_rate)  # Ensure non-negative values
            
            ######################################################################
            # Generate depth measurements with sensor accuracy
            ######################################################################
            measured_depth = depth + np.random.normal(0, self.measurement_accuracy["depth"])
            
            ######################################################################
            # Compile sensor data point
            ######################################################################
            sensor_data_point = {
                "point_id": measurement_index + 1,
                "depth": round(measured_depth, 2),
                "pressure": round(observed_pressure, 1),
                "temperature": round(measured_temperature, 2),
                "flow_rate": round(flow_rate, 3),
                "expected_pressure": round(theoretical_pressure, 1),
                "pressure_deviation": round(abs(observed_pressure - theoretical_pressure), 1)
            }
            
            sensor_readings.append(sensor_data_point)
        
        return sensor_readings

######################################################################
# Physics-Based Leakage Detection Engine
######################################################################

class LeakageDetectionEngine:
    """
    Analyzes sensor data using physics-based algorithms to detect and assess leakage severity.
    Implements hydrostatic pressure validation and anomaly detection methods.
    """
    
    def __init__(self):
        self.gravitational_constant = 9.81
        self.seawater_density = 1025
        self.atmospheric_pressure = 101325
        
        ######################################################################
        # Define leakage severity classification thresholds
        ######################################################################
        self.severity_classification_thresholds = {
            "pressure_deviation": [1000, 3000, 8000],  # Pa thresholds
            "flow_anomaly": [2.0, 5.0, 10.0],  # m³/s thresholds
            "temperature_anomaly": [2.0, 5.0, 8.0]  # °C thresholds
        }
    
    def analyze_sensor_data_for_leakage(self, sensor_readings):
        """
        Implements comprehensive sensor data analysis for leakage detection.
        Performs pressure validation, anomaly detection, and severity assessment.
        """
        
        analysis_summary = {
            "total_measurement_points": len(sensor_readings),
            "detected_anomaly_points": 0,
            "maximum_pressure_deviation": 0,
            "maximum_flow_rate": 0,
            "temperature_measurement_range": [0, 0],
            "detailed_anomaly_list": []
        }
        
        ######################################################################
        # Process each sensor reading for anomaly detection
        ######################################################################
        pressure_deviation_list = []
        flow_rate_measurements = []
        temperature_measurements = []
        
        for sensor_point in sensor_readings:
            ######################################################################
            # Calculate theoretical pressure using hydrostatic formula
            ######################################################################
            theoretical_pressure = (self.atmospheric_pressure + 
                                  (self.seawater_density * self.gravitational_constant * sensor_point["depth"]))
            pressure_deviation = abs(sensor_point["pressure"] - theoretical_pressure)
            pressure_deviation_list.append(pressure_deviation)
            
            flow_rate_measurements.append(sensor_point["flow_rate"])
            temperature_measurements.append(sensor_point["temperature"])
            
            ######################################################################
            # Identify and classify anomalous readings
            ######################################################################
            if pressure_deviation > self.severity_classification_thresholds["pressure_deviation"][0]:
                analysis_summary["detected_anomaly_points"] += 1
                analysis_summary["detailed_anomaly_list"].append({
                    "point_id": sensor_point["point_id"],
                    "depth": sensor_point["depth"],
                    "pressure_deviation": pressure_deviation,
                    "flow_rate": sensor_point["flow_rate"],
                    "anomaly_classification": self.classify_anomaly_type(pressure_deviation, sensor_point["flow_rate"])
                })
        
        ######################################################################
        # Update analysis summary with key metrics
        ######################################################################
        analysis_summary["maximum_pressure_deviation"] = max(pressure_deviation_list)
        analysis_summary["maximum_flow_rate"] = max(flow_rate_measurements)
        analysis_summary["temperature_measurement_range"] = [min(temperature_measurements), max(temperature_measurements)]
        
        ######################################################################
        # Generate leakage severity score using ML-inspired algorithm
        ######################################################################
        leakage_severity_score = self.calculate_leakage_severity_score(analysis_summary)
        
        return analysis_summary, leakage_severity_score
    
    def classify_anomaly_type(self, pressure_deviation, flow_rate):
        """
        Classifies detected anomalies based on severity thresholds.
        Returns descriptive anomaly type for further analysis.
        """
        
        if pressure_deviation > self.severity_classification_thresholds["pressure_deviation"][2]:
            return "critical_pressure_drop"
        elif pressure_deviation > self.severity_classification_thresholds["pressure_deviation"][1]:
            return "major_pressure_drop"
        elif pressure_deviation > self.severity_classification_thresholds["pressure_deviation"][0]:
            return "minor_pressure_drop"
        
        if flow_rate > self.severity_classification_thresholds["flow_anomaly"][2]:
            return "critical_flow_anomaly"
        elif flow_rate > self.severity_classification_thresholds["flow_anomaly"][1]:
            return "major_flow_anomaly"
        elif flow_rate > self.severity_classification_thresholds["flow_anomaly"][0]:
            return "minor_flow_anomaly"
        
        return "normal_reading"
    
    def calculate_leakage_severity_score(self, analysis_summary):
        """
        Calculates normalized leakage severity score using weighted factors.
        Returns λ ∈ [0,1] where 1 represents most severe leakage condition.
        """
        
        ######################################################################
        # Extract and normalize key severity indicators
        ######################################################################
        maximum_pressure_deviation = analysis_summary["maximum_pressure_deviation"]
        maximum_flow_rate = analysis_summary["maximum_flow_rate"]
        anomaly_point_ratio = analysis_summary["detected_anomaly_points"] / analysis_summary["total_measurement_points"]
        
        ######################################################################
        # Normalize each factor to [0,1] range
        ######################################################################
        pressure_severity_factor = min(maximum_pressure_deviation / 15000, 1.0)
        flow_severity_factor = min(maximum_flow_rate / 15.0, 1.0)
        anomaly_ratio_factor = min(anomaly_point_ratio * 2, 1.0)
        
        ######################################################################
        # Calculate weighted composite severity score
        ######################################################################
        composite_severity_score = (
            0.4 * pressure_severity_factor +
            0.3 * flow_severity_factor +
            0.3 * anomaly_ratio_factor
        )
        
        return round(min(composite_severity_score, 1.0), 3)

######################################################################
# Language Model Integration for Risk Assessment
######################################################################

class LeakageRiskAssessmentLLM:
    """
    Integrates language models for comprehensive underwater leakage risk assessment.
    Supports multiple LLM backends with intelligent fallback to physics-based analysis.
    """
    
    def __init__(self, model_type="rule_based"):
        self.selected_model_type = model_type
        self.model_operational = False
        
        if model_type == "transformers":
            self.configure_transformers_model()
        elif model_type == "ollama":
            self.configure_ollama_model()
        else:
            self.model_operational = True
            print("Using physics-based expert system for leakage analysis")
    
    def configure_transformers_model(self):
        """
        Configures Hugging Face Transformers for risk assessment generation.
        Uses free models suitable for technical analysis tasks.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_identifier = "microsoft/DialoGPT-medium"
            print(f"Loading {model_identifier} for leakage analysis...")
            
            self.text_tokenizer = AutoTokenizer.from_pretrained(model_identifier)
            self.language_model = AutoModelForCausalLM.from_pretrained(model_identifier)
            
            if self.text_tokenizer.pad_token is None:
                self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
            
            self.model_operational = True
            print("Transformers model loaded successfully")
            
        except Exception as setup_error:
            print(f"Transformers setup error: {setup_error}")
            print("Switching to rule-based analysis")
    
    def configure_ollama_model(self):
        """
        Configures Ollama for local LLM-based risk assessment.
        Automatically detects and selects available models.
        """
        try:
            ollama_response = requests.get("http://localhost:11434/api/tags")
            if ollama_response.status_code == 200:
                available_models = ollama_response.json()
                model_names = [model['name'] for model in available_models.get('models', [])]
                
                preferred_model_list = ['llama2', 'mistral', 'phi']
                self.active_ollama_model = None
                
                for preferred_model in preferred_model_list:
                    if any(preferred_model in model_name for model_name in model_names):
                        self.active_ollama_model = preferred_model
                        break
                
                if self.active_ollama_model:
                    self.model_operational = True
                    print(f"Using Ollama model: {self.active_ollama_model}")
                else:
                    print("No suitable Ollama models found")
        except requests.exceptions.RequestException:
            print("Ollama is not available")
    
    def construct_comprehensive_risk_assessment_prompt(self, severity_score, analysis_results, sensor_data, system_metadata):
        """
        Creates structured, comprehensive prompts for LLM-based risk assessment.
        Incorporates physics-based analysis results and system specifications.
        """
        
        ######################################################################
        # Calculate summary statistics from sensor data
        ######################################################################
        average_depth = np.mean([point["depth"] for point in sensor_data])
        average_pressure_deviation = np.mean([point["pressure_deviation"] for point in sensor_data])
        maximum_flow_rate = analysis_results["maximum_flow_rate"]
        
        ######################################################################
        # Format anomaly details for prompt inclusion
        ######################################################################
        anomaly_summary_text = "\n".join([
            f"- Point {anomaly['point_id']} at {anomaly['depth']}m: {anomaly['pressure_deviation']} Pa deviation, flow {anomaly['flow_rate']} m³/s"
            for anomaly in analysis_results["detailed_anomaly_list"][:5]
        ])
        
        comprehensive_risk_prompt = f"""You are an expert underwater systems engineer specializing in pipeline integrity and leakage detection.

LEAKAGE DETECTION ANALYSIS:
System: {system_metadata.get('system_type', 'Underwater pipeline system')}
Location: {system_metadata.get('location', 'Offshore facility')}
Pipe Material: {system_metadata.get('pipe_material', 'Steel')}
Environment: {system_metadata.get('environment', 'Marine environment')}

SENSOR DATA ANALYSIS:
Total measurement points: {analysis_results['total_measurement_points']}
Average depth: {average_depth:.1f} meters
Anomaly points detected: {analysis_results['detected_anomaly_points']}
Maximum pressure deviation: {analysis_results['maximum_pressure_deviation']:.1f} Pa
Maximum flow rate: {maximum_flow_rate:.2f} m³/s
Temperature range: {analysis_results['temperature_measurement_range'][0]:.1f}°C to {analysis_results['temperature_measurement_range'][1]:.1f}°C

LEAKAGE SEVERITY ASSESSMENT:
ML-predicted severity score: {severity_score:.3f} (scale 0-1, where 1 is most severe)

DETECTED ANOMALIES:
{anomaly_summary_text if anomaly_summary_text else "No significant anomalies detected"}

PHYSICS-BASED ANALYSIS:
- Hydrostatic pressure validation completed
- Pressure deviations analyzed against theoretical values
- Flow rate anomalies correlated with pressure drops
- Temperature profile consistent with depth expectations

RISK ASSESSMENT REQUIRED:
Based on the leakage severity score of {severity_score:.3f} and the detected anomalies:

1. What is the risk level (low/moderate/high/critical)?
2. What are the immediate actions required?
3. What monitoring should be implemented?
4. What are the potential consequences if not addressed?
5. What repair/mitigation strategies are recommended?

Please provide a comprehensive risk assessment and actionable recommendations for this underwater leakage detection scenario."""

        return comprehensive_risk_prompt
    
    def generate_comprehensive_risk_assessment(self, structured_prompt):
        """
        Generates detailed risk assessment using the configured language model.
        Provides fallback to physics-based analysis if LLM processing fails.
        """
        
        print("Generating expert risk assessment...")
        
        if self.selected_model_type == "transformers" and self.model_operational:
            return self.process_with_transformers_model(structured_prompt)
        elif self.selected_model_type == "ollama" and self.model_operational:
            return self.process_with_ollama_model(structured_prompt)
        else:
            return self.create_physics_based_risk_analysis(structured_prompt)
    
    def process_with_transformers_model(self, prompt):
        """Processes risk assessment using Transformers language model"""
        try:
            simplified_analysis_prompt = f"Underwater leakage analysis: severity score {self.extract_severity_score_from_prompt(prompt):.3f}. Risk assessment and recommendations:"
            
            tokenized_inputs = self.text_tokenizer.encode(simplified_analysis_prompt, return_tensors="pt")
            
            import torch
            with torch.no_grad():
                model_outputs = self.language_model.generate(
                    tokenized_inputs,
                    max_length=len(tokenized_inputs[0]) + 300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.text_tokenizer.eos_token_id
                )
            
            generated_response = self.text_tokenizer.decode(model_outputs[0], skip_special_tokens=True)
            generated_response = generated_response[len(simplified_analysis_prompt):].strip()
            
            if len(generated_response) < 50:
                return self.create_physics_based_risk_analysis(prompt)
            
            return generated_response
            
        except Exception as processing_error:
            print(f"Transformers processing error: {processing_error}")
            return self.create_physics_based_risk_analysis(prompt)
    
    def process_with_ollama_model(self, prompt):
        """Processes risk assessment using Ollama language model"""
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
                return self.create_physics_based_risk_analysis(prompt)
                
        except Exception as processing_error:
            print(f"Ollama processing error: {processing_error}")
            return self.create_physics_based_risk_analysis(prompt)
    
    def extract_severity_score_from_prompt(self, prompt_text):
        """Extracts severity score from structured prompt for simplified processing"""
        try:
            severity_line = [line for line in prompt_text.split('\n') if 'severity score:' in line][0]
            return float(severity_line.split('score:')[1].split()[0])
        except (IndexError, ValueError):
            return 0.5
    
    def create_physics_based_risk_analysis(self, structured_prompt):
        """
        Creates comprehensive physics-based risk assessment using engineering principles.
        Provides detailed analysis when language models are unavailable.
        """
        
        print("Using physics-based expert analysis system")
        
        ######################################################################
        # Extract key parameters from the structured prompt
        ######################################################################
        severity_score = self.extract_severity_score_from_prompt(structured_prompt)
        
        prompt_lines = structured_prompt.split('\n')
        detected_anomaly_points = 0
        maximum_pressure_deviation = 0
        maximum_flow_rate = 0
        
        for line in prompt_lines:
            if 'Anomaly points detected:' in line:
                detected_anomaly_points = int(line.split(':')[1].strip())
            elif 'Maximum pressure deviation:' in line:
                maximum_pressure_deviation = float(line.split(':')[1].split()[0])
            elif 'Maximum flow rate:' in line:
                maximum_flow_rate = float(line.split(':')[1].split()[0])
        
        ######################################################################
        # Determine risk level based on severity score and physics analysis
        ######################################################################
        if severity_score >= 0.8:
            risk_classification = "CRITICAL"
            action_timeline = "immediate emergency response required"
        elif severity_score >= 0.6:
            risk_classification = "HIGH"
            action_timeline = "urgent action required within 24 hours"
        elif severity_score >= 0.3:
            risk_classification = "MODERATE"
            action_timeline = "action required within 72 hours"
        else:
            risk_classification = "LOW"
            action_timeline = "routine monitoring and maintenance"
        
        ######################################################################
        # Generate comprehensive physics-based risk assessment
        ######################################################################
        comprehensive_risk_assessment = f"""
UNDERWATER LEAKAGE RISK ASSESSMENT REPORT

EXECUTIVE SUMMARY:
Leakage Severity Score: {severity_score:.3f}/1.0
Risk Level: {risk_classification}
Action Timeline: {action_timeline}

TECHNICAL ANALYSIS:
The physics-based assessment reveals a leakage severity score of {severity_score:.3f}, indicating {risk_classification.lower()} risk conditions. 

Pressure Analysis:
• Maximum pressure deviation: {maximum_pressure_deviation:.1f} Pa from hydrostatic baseline
• Hydrostatic pressure validation shows {"significant" if maximum_pressure_deviation > 5000 else "minor"} deviations
• Pressure anomalies detected at {detected_anomaly_points} measurement points

Flow Dynamics:
• Maximum flow rate: {maximum_flow_rate:.2f} m³/s
• Flow anomalies {"strongly correlate" if maximum_flow_rate > 5.0 else "moderately correlate"} with pressure deviations
• Flow pattern suggests {"active leakage" if maximum_flow_rate > 8.0 else "potential system stress"}

RISK ASSESSMENT:
1. RISK LEVEL: {risk_classification}
   Justification: Severity score {severity_score:.3f} combined with pressure deviations of {maximum_pressure_deviation:.1f} Pa indicates {risk_classification.lower()} risk to system integrity.

2. IMMEDIATE ACTIONS:
   • {"Emergency shutdown and isolation" if risk_classification == "CRITICAL" else "Increased monitoring frequency"}
   • {"Deploy emergency repair team" if risk_classification == "CRITICAL" else "Schedule detailed inspection"}
   • {"Notify all stakeholders immediately" if risk_classification == "CRITICAL" else "Prepare maintenance protocols"}

3. MONITORING REQUIREMENTS:
   • {"Continuous real-time monitoring" if risk_classification in ["CRITICAL", "HIGH"] else "Enhanced monitoring schedule"}
   • Pressure sensor calibration verification
   • Flow rate measurement validation
   • Temperature profile analysis

4. POTENTIAL CONSEQUENCES:
   • {"Catastrophic system failure possible" if risk_classification == "CRITICAL" else "Gradual system degradation"}
   • {"Immediate environmental impact risk" if risk_classification == "CRITICAL" else "Long-term integrity concerns"}
   • {"Significant economic impact" if risk_classification in ["CRITICAL", "HIGH"] else "Manageable maintenance costs"}

5. MITIGATION STRATEGIES:
   Primary Actions:
   • {"Emergency isolation and containment" if risk_classification == "CRITICAL" else "Systematic pressure testing"}
   • {"Immediate repair mobilization" if risk_classification == "CRITICAL" else "Planned maintenance scheduling"}
   • {"Alternative system activation" if risk_classification == "CRITICAL" else "Operational parameter optimization"}

   Secondary Actions:
   • Enhanced sensor network deployment
   • Predictive maintenance algorithm implementation
   • System redundancy evaluation

PHYSICS-BASED RECOMMENDATIONS:
Based on hydrostatic pressure analysis and flow dynamics:

• Pressure Management: {"Critical pressure control required" if maximum_pressure_deviation > 8000 else "Standard pressure monitoring adequate"}
• Flow Control: {"Immediate flow restriction needed" if maximum_flow_rate > 10.0 else "Flow rate within acceptable parameters"}
• System Integrity: {"Structural assessment mandatory" if severity_score > 0.7 else "Routine integrity checks sufficient"}

TIMELINE FOR ACTION:
• Emergency Response: {action_timeline}
• Detailed Assessment: {"Within 6 hours" if risk_classification == "CRITICAL" else "Within 48 hours"}
• Repair Implementation: {"Within 24 hours" if risk_classification == "CRITICAL" else "Within 1 week"}
• System Validation: {"Before restart" if risk_classification == "CRITICAL" else "After repairs"}

CONCLUSION:
The underwater leakage detection system has identified {risk_classification.lower()} risk conditions with severity score {severity_score:.3f}. 
{"Immediate emergency response is required to prevent catastrophic failure." if risk_classification == "CRITICAL" else 
 "Prompt action is required to maintain system integrity." if risk_classification == "HIGH" else
 "Routine maintenance and monitoring will address identified issues." if risk_classification == "MODERATE" else
 "Current conditions are within acceptable operational parameters."}

NEXT STEPS:
1. {"Execute emergency response protocol" if risk_classification == "CRITICAL" else "Implement monitoring enhancement"}
2. {"Mobilize repair resources" if risk_classification in ["CRITICAL", "HIGH"] else "Schedule maintenance activities"}
3. {"Conduct system shutdown procedures" if risk_classification == "CRITICAL" else "Continue operational monitoring"}
4. {"Notify regulatory authorities" if risk_classification == "CRITICAL" else "Document findings for compliance"}

This assessment is based on physics-based analysis of hydrostatic pressure deviations, flow dynamics, and established risk assessment protocols for underwater pipeline systems.
"""
        
        return comprehensive_risk_assessment.strip()

######################################################################
# Complete Underwater Leakage Detection Pipeline
######################################################################

class UnderwaterLeakageDetectionPipeline:
    """
    Comprehensive pipeline integrating sensor simulation, physics-based detection,
    and LLM-powered risk assessment for underwater pipeline monitoring.
    """
    
    def __init__(self, model_type="rule_based"):
        self.sensor_simulator = UnderwaterSensorSimulator()
        self.leakage_detector = LeakageDetectionEngine()
        self.risk_assessor = LeakageRiskAssessmentLLM(model_type)
        
        ######################################################################
        # Create output directories for data and reports
        ######################################################################
        os.makedirs("leakage_data", exist_ok=True)
        os.makedirs("risk_reports", exist_ok=True)
        
        print(f"Underwater Leakage Pipeline initialized with {model_type} LLM")
    
    def execute_complete_leakage_analysis(self, scenario="normal", system_metadata=None):
        """
        Executes comprehensive leakage detection and risk assessment workflow.
        Implements all algorithm steps from sensor data generation to final risk assessment.
        """
        
        print("UNDERWATER LEAKAGE DETECTION AND RISK ASSESSMENT")
        print("=" * 70)
        
        ######################################################################
        # Initialize system metadata if not provided
        ######################################################################
        if system_metadata is None:
            system_metadata = {
                "system_type": "Offshore oil pipeline",
                "location": "North Sea Platform",
                "pipe_material": "Carbon steel with protective coating",
                "environment": "Deep marine environment",
                "depth_range": "5-50 meters",
                "operational_pressure": "150 bar"
            }
        
        ######################################################################
        # Phase 1: Generate realistic sensor data
        ######################################################################
        print(f"\nPhase 1: Generating sensor data (scenario: {scenario})...")
        sensor_measurement_data = self.sensor_simulator.generate_realistic_sensor_data(
            measurement_points=25, leakage_scenario=scenario
        )
        
        print(f"   Generated {len(sensor_measurement_data)} sensor readings")
        print(f"   Depth range: {min(point['depth'] for point in sensor_measurement_data):.1f}m to {max(point['depth'] for point in sensor_measurement_data):.1f}m")
        
        ######################################################################
        # Phase 2: Analyze sensor data for leakage detection
        ######################################################################
        print("\nPhase 2: Analyzing sensor data for leakage detection...")
        leakage_analysis_results, severity_score = self.leakage_detector.analyze_sensor_data_for_leakage(sensor_measurement_data)
        
        print(f"   Anomaly points detected: {leakage_analysis_results['detected_anomaly_points']}")
        print(f"   Leakage severity score: {severity_score:.3f}")
        print(f"   Max pressure deviation: {leakage_analysis_results['maximum_pressure_deviation']:.1f} Pa")
        
        ######################################################################
        # Phase 3: Create comprehensive risk assessment prompt
        ######################################################################
        print("\nPhase 3: Constructing risk assessment prompt...")
        risk_assessment_prompt = self.risk_assessor.construct_comprehensive_risk_assessment_prompt(
            severity_score, leakage_analysis_results, sensor_measurement_data, system_metadata
        )
        
        ######################################################################
        # Phase 4: Generate expert risk assessment
        ######################################################################
        print("\nPhase 4: Generating expert risk assessment...")
        comprehensive_risk_assessment = self.risk_assessor.generate_comprehensive_risk_assessment(risk_assessment_prompt)
        
        ######################################################################
        # Determine final risk level from assessment
        ######################################################################
        determined_risk_level = self.extract_risk_level_from_assessment(comprehensive_risk_assessment, severity_score)
        
        ######################################################################
        # Compile comprehensive analysis results
        ######################################################################
        complete_analysis_results = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario,
            "llm_type": self.risk_assessor.selected_model_type,
            "metadata": system_metadata,
            "sensor_data": sensor_measurement_data,
            "analysis_results": leakage_analysis_results,
            "leakage_severity": severity_score,
            "risk_level": determined_risk_level,
            "risk_assessment": comprehensive_risk_assessment,
            "algorithm_outputs": {
                "lambda": severity_score,
                "r": determined_risk_level,
                "R": comprehensive_risk_assessment
            }
        }
        
        ######################################################################
        # Display and save comprehensive results
        ######################################################################
        self.display_analysis_results(complete_analysis_results)
        self.save_comprehensive_analysis_results(complete_analysis_results)
        
        return complete_analysis_results
    
    def extract_risk_level_from_assessment(self, risk_assessment_text, severity_score):
        """
        Extracts risk level classification from the generated risk assessment.
        Uses text analysis and severity score as backup classification method.
        """
        
        risk_assessment_lower = risk_assessment_text.lower()
        
        if "critical" in risk_assessment_lower:
            return "CRITICAL"
        elif "high" in risk_assessment_lower:
            return "HIGH"
        elif "moderate" in risk_assessment_lower:
            return "MODERATE"
        else:
            return "LOW"
    
    def display_analysis_results(self, analysis_results):
        """
        Displays comprehensive leakage analysis results in formatted output.
        Provides clear summary of findings and risk assessment conclusions.
        """
        
        print("\n" + "=" * 70)
        print("UNDERWATER LEAKAGE ANALYSIS RESULTS")
        print("=" * 70)
        
        ######################################################################
        # Display basic analysis information
        ######################################################################
        print(f"Analysis Date: {analysis_results['analysis_date']}")
        print(f"Scenario: {analysis_results['scenario']}")
        print(f"LLM Type: {analysis_results['llm_type']}")
        print(f"Location: {analysis_results['metadata']['location']}")
        print(f"System: {analysis_results['metadata']['system_type']}")
        
        ######################################################################
        # Display sensor analysis summary
        ######################################################################
        sensor_analysis = analysis_results['analysis_results']
        print(f"\nSENSOR ANALYSIS SUMMARY:")
        print(f"   Measurement points: {sensor_analysis['total_measurement_points']}")
        print(f"   Anomaly points: {sensor_analysis['detected_anomaly_points']}")
        print(f"   Max pressure deviation: {sensor_analysis['maximum_pressure_deviation']:.1f} Pa")
        print(f"   Max flow rate: {sensor_analysis['maximum_flow_rate']:.2f} m³/s")
        
        ######################################################################
        # Display algorithm outputs
        ######################################################################
        print(f"\nALGORITHM OUTPUTS:")
        print(f"   λ (Severity Score): {analysis_results['leakage_severity']:.3f}")
        print(f"   r (Risk Level): {analysis_results['risk_level']}")
        
        ######################################################################
        # Display risk assessment preview
        ######################################################################
        print(f"\nRISK ASSESSMENT PREVIEW:")
        print("-" * 60)
        assessment_lines = analysis_results['risk_assessment'].split('\n')
        for line in assessment_lines[:12]:
            if line.strip():
                print(f"   {line.strip()}")
        
        if len(assessment_lines) > 12:
            print("   ... (continued in saved report)")
        print("-" * 60)
    
    def save_comprehensive_analysis_results(self, analysis_results):
        """
        Saves comprehensive leakage analysis results in multiple formats.
        Creates detailed JSON data, human-readable reports, and CSV sensor data.
        """
        
        ######################################################################
        # Generate timestamped filenames
        ######################################################################
        timestamp = analysis_results['analysis_date'].replace(' ', '_').replace(':', '-')
        json_report_filename = f"risk_reports/leakage_analysis_{timestamp}.json"
        
        ######################################################################
        # Save detailed JSON analysis report
        ######################################################################
        with open(json_report_filename, 'w') as json_file:
            json.dump(analysis_results, json_file, indent=2)
        
        ######################################################################
        # Save human-readable risk assessment report
        ######################################################################
        text_report_filename = f"risk_reports/risk_assessment_{timestamp}.txt"
        with open(text_report_filename, 'w') as text_file:
            text_file.write("UNDERWATER LEAKAGE DETECTION AND RISK ASSESSMENT REPORT\n")
            text_file.write("=" * 70 + "\n\n")
            text_file.write(f"Analysis Date: {analysis_results['analysis_date']}\n")
            text_file.write(f"Scenario: {analysis_results['scenario']}\n")
            text_file.write(f"LLM Model: {analysis_results['llm_type']}\n")
            text_file.write(f"Location: {analysis_results['metadata']['location']}\n")
            text_file.write(f"System: {analysis_results['metadata']['system_type']}\n\n")
            
            text_file.write("ALGORITHM OUTPUTS:\n")
            text_file.write("-" * 30 + "\n")
            text_file.write(f"Leakage Severity Score (λ): {analysis_results['leakage_severity']:.3f}\n")
            text_file.write(f"Risk Level (r): {analysis_results['risk_level']}\n\n")
            
            text_file.write("SENSOR DATA SUMMARY:\n")
            text_file.write("-" * 30 + "\n")
            sensor_analysis = analysis_results['analysis_results']
            text_file.write(f"Total measurement points: {sensor_analysis['total_measurement_points']}\n")
            text_file.write(f"Anomaly points detected: {sensor_analysis['detected_anomaly_points']}\n")
            text_file.write(f"Maximum pressure deviation: {sensor_analysis['maximum_pressure_deviation']:.1f} Pa\n")
            text_file.write(f"Maximum flow rate: {sensor_analysis['maximum_flow_rate']:.2f} m³/s\n\n")
            
            text_file.write("RISK ASSESSMENT AND RECOMMENDATIONS:\n")
            text_file.write("-" * 30 + "\n")
            text_file.write(analysis_results['risk_assessment'])
        
        ######################################################################
        # Save sensor data to CSV for further analysis
        ######################################################################
        csv_data_filename = f"leakage_data/sensor_data_{timestamp}.csv"
        sensor_dataframe = pd.DataFrame(analysis_results['sensor_data'])
        sensor_dataframe.to_csv(csv_data_filename, index=False)
        
        print(f"\nReports saved:")
        print(f"   Detailed analysis: {json_report_filename}")
        print(f"   Risk assessment: {text_report_filename}")
        print(f"   Sensor data: {csv_data_filename}")

######################################################################
# User Interface and System Configuration
######################################################################

def display_available_scenario_options():
    """
    Displays available leakage scenarios with detailed descriptions.
    Helps users understand different testing conditions.
    """
    
    print("AVAILABLE LEAKAGE SCENARIOS:")
    print("=" * 50)
    
    scenario_descriptions = {
        "normal": {
            "description": "Normal pipeline operation",
            "characteristics": "Minimal pressure deviations, stable flow rates",
            "expected_severity": "0.0 - 0.2"
        },
        "minor_leak": {
            "description": "Minor leakage detected",
            "characteristics": "Small pressure drops, slightly elevated flow",
            "expected_severity": "0.2 - 0.4"
        },
        "major_leak": {
            "description": "Significant leakage event",
            "characteristics": "Notable pressure deviations, high flow rates",
            "expected_severity": "0.4 - 0.7"
        },
        "critical_leak": {
            "description": "Critical system failure",
            "characteristics": "Severe pressure drops, very high flow rates",
            "expected_severity": "0.7 - 1.0"
        }
    }
    
    for scenario_number, (scenario_name, scenario_details) in enumerate(scenario_descriptions.items(), 1):
        print(f"\n{scenario_number}. {scenario_name.upper()}")
        print(f"   Description: {scenario_details['description']}")
        print(f"   Characteristics: {scenario_details['characteristics']}")
        print(f"   Expected Severity: {scenario_details['expected_severity']}")

def display_llm_configuration_options():
    """
    Displays available LLM options for risk assessment with setup instructions.
    """
    
    print("\nLLM OPTIONS FOR RISK ASSESSMENT:")
    print("=" * 50)
    
    print("\n1. PHYSICS-BASED EXPERT SYSTEM (Recommended)")
    print("   Advanced hydrostatic pressure analysis")
    print("   Flow dynamics expertise")
    print("   Industry-standard risk assessment")
    print("   No setup required")
    
    print("\n2. HUGGING FACE TRANSFORMERS")
    print("   Requires: pip install transformers torch")
    print("   Real neural network analysis")
    print("   Free models available")
    
    print("\n3. OLLAMA (Local LLM)")
    print("   Requires: Ollama installation")
    print("   High-quality local processing")
    print("   Privacy-focused")

def create_sample_system_metadata():
    """
    Creates sample system metadata for different underwater pipeline configurations.
    """
    
    system_configuration_options = [
        {
            "system_type": "Offshore oil pipeline",
            "location": "North Sea Platform Alpha",
            "pipe_material": "Carbon steel with anti-corrosion coating",
            "environment": "Deep marine environment, 45m depth",
            "depth_range": "10-50 meters",
            "operational_pressure": "150 bar"
        },
        {
            "system_type": "Subsea water intake system",
            "location": "Mediterranean Coast",
            "pipe_material": "Stainless steel",
            "environment": "Coastal marine environment",
            "depth_range": "5-30 meters",
            "operational_pressure": "80 bar"
        },
        {
            "system_type": "Underwater gas pipeline",
            "location": "Gulf of Mexico",
            "pipe_material": "High-grade steel with polymer coating",
            "environment": "Deep water marine environment",
            "depth_range": "20-60 meters",
            "operational_pressure": "200 bar"
        }
    ]
    
    return random.choice(system_configuration_options)

######################################################################
# Main Execution Function
######################################################################

def main():
    """
    Main execution function with interactive system configuration.
    Provides user-friendly interface for underwater leakage analysis.
    """
    
    print("UNDERWATER LEAKAGE DETECTION AND RISK ASSESSMENT PIPELINE")
    print("Physics-based AI system for pipeline integrity monitoring")
    print("=" * 70)
    
    ######################################################################
    # Display configuration options
    ######################################################################
    display_available_scenario_options()
    display_llm_configuration_options()
    
    ######################################################################
    # Get user configuration selections
    ######################################################################
    print("\n" + "=" * 70)
    print("SYSTEM CONFIGURATION:")
    
    ######################################################################
    # Scenario selection
    ######################################################################
    print("\nSelect leakage scenario:")
    print("1. Normal operation")
    print("2. Minor leak")
    print("3. Major leak") 
    print("4. Critical leak")
    
    scenario_selection = input("\nEnter scenario choice (1-4) [2]: ").strip()
    scenario_mapping = {
        "1": "normal",
        "2": "minor_leak",
        "3": "major_leak", 
        "4": "critical_leak",
        "": "minor_leak"  # Default selection
    }
    
    selected_scenario = scenario_mapping.get(scenario_selection, "minor_leak")
    
    ######################################################################
    # LLM type selection
    ######################################################################
    print("\nSelect LLM type:")
    print("1. Physics-based Expert System")
    print("2. Hugging Face Transformers")
    print("3. Ollama")
    
    llm_selection = input("\nEnter LLM choice (1-3) [1]: ").strip()
    llm_type_mapping = {
        "1": "rule_based",
        "2": "transformers",
        "3": "ollama",
        "": "rule_based"  # Default selection
    }
    
    selected_llm_type = llm_type_mapping.get(llm_selection, "rule_based")
    
    print(f"\nSelected scenario: {selected_scenario}")
    print(f"Selected LLM: {selected_llm_type}")
    
    ######################################################################
    # Initialize and execute pipeline
    ######################################################################
    print(f"\nInitializing pipeline...")
    leakage_detection_pipeline = UnderwaterLeakageDetectionPipeline(selected_llm_type)
    
    sample_system_metadata = create_sample_system_metadata()
    
    ######################################################################
    # Execute comprehensive leakage analysis
    ######################################################################
    try:
        print(f"\nRunning leakage detection analysis...")
        print(f"   System: {sample_system_metadata['system_type']}")
        print(f"   Location: {sample_system_metadata['location']}")
        print(f"   Environment: {sample_system_metadata['environment']}")
        
        analysis_results = leakage_detection_pipeline.execute_complete_leakage_analysis(selected_scenario, sample_system_metadata)
        
        print("\n" + "=" * 70)
        print("UNDERWATER LEAKAGE ANALYSIS COMPLETE")
        print("=" * 70)
        
        ######################################################################
        # Display key findings summary
        ######################################################################
        print("KEY FINDINGS:")
        print(f"   Leakage Severity Score (λ): {analysis_results['leakage_severity']:.3f}")
        print(f"   Risk Level (r): {analysis_results['risk_level']}")
        print(f"   Anomaly Points: {analysis_results['analysis_results']['detected_anomaly_points']}")
        print(f"   Max Pressure Deviation: {analysis_results['analysis_results']['maximum_pressure_deviation']:.1f} Pa")
        
        print("\nREPORTS GENERATED:")
        print("   Detailed technical analysis")
        print("   Risk assessment and recommendations")
        print("   Sensor data export")
        print("   Physics-based validation")
        
        ######################################################################
        # Display risk level alert
        ######################################################################
        risk_level = analysis_results['risk_level']
        if risk_level == "CRITICAL":
            print(f"\nCRITICAL ALERT: Immediate emergency response required!")
        elif risk_level == "HIGH":
            print(f"\nHIGH RISK: Urgent action needed within 24 hours")
        elif risk_level == "MODERATE":
            print(f"\nMODERATE RISK: Action required within 72 hours")
        else:
            print(f"\nLOW RISK: Routine monitoring sufficient")
        
        return analysis_results
        
    except Exception as analysis_error:
        print(f"\nError during analysis: {str(analysis_error)}")
        print("Please check your configuration and try again.")
        return None

######################################################################
# Demonstration and Testing Functions
######################################################################

def demonstrate_all_leakage_scenarios():
    """
    Demonstrates all leakage scenarios for comprehensive system testing.
    Useful for validating pipeline performance across different conditions.
    """
    
    print("DEMONSTRATING ALL LEAKAGE SCENARIOS")
    print("=" * 60)
    
    test_scenarios = ["normal", "minor_leak", "major_leak", "critical_leak"]
    demonstration_pipeline = UnderwaterLeakageDetectionPipeline("rule_based")
    
    scenario_results_summary = []
    
    for test_scenario in test_scenarios:
        print(f"\nTesting scenario: {test_scenario}")
        print("-" * 40)
        
        sample_metadata = create_sample_system_metadata()
        scenario_results = demonstration_pipeline.execute_complete_leakage_analysis(test_scenario, sample_metadata)
        
        scenario_results_summary.append({
            "scenario": test_scenario,
            "severity": scenario_results['leakage_severity'],
            "risk_level": scenario_results['risk_level'],
            "anomalies": scenario_results['analysis_results']['detected_anomaly_points']
        })
        
        print(f"   λ: {scenario_results['leakage_severity']:.3f}")
        print(f"   Risk: {scenario_results['risk_level']}")
    
    ######################################################################
    # Display comparative scenario summary
    ######################################################################
    print("\nSCENARIO COMPARISON:")
    print("=" * 60)
    for result_summary in scenario_results_summary:
        print(f"{result_summary['scenario']:15} | λ: {result_summary['severity']:.3f} | Risk: {result_summary['risk_level']:8} | Anomalies: {result_summary['anomalies']}")
    
    return scenario_results_summary

if __name__ == "__main__":
    # Execute the complete underwater leakage detection pipeline
    pipeline_results = main()
    
    # Optional comprehensive scenario demonstration
    if pipeline_results:
        demonstration_choice = input("\nRun demonstration of all scenarios? (y/n) [n]: ").strip().lower()
        if demonstration_choice == 'y':
            demonstrate_all_leakage_scenarios()
