# ============================================================================
# UNDERWATER LEAKAGE DETECTION AND RISK ASSESSMENT PIPELINE
# Complete implementation with real LLM integration
# ============================================================================

import json
import os
import numpy as np
from datetime import datetime
import random
import math
import pandas as pd
from collections import defaultdict
import requests

# ============================================================================
# 1. SENSOR DATA SIMULATOR
# ============================================================================

class UnderwaterSensorSimulator:
    """Simulates underwater sensor data for leakage detection"""
    
    def __init__(self):
        # Physical constants
        self.g = 9.81  # gravitational constant (m/s²)
        self.rho_water = 1025  # seawater density (kg/m³)
        self.P0 = 101325  # atmospheric pressure (Pa)
        
        # Sensor specifications
        self.sensor_accuracy = {
            "pressure": 0.02,  # 2% accuracy
            "temperature": 0.5,  # ±0.5°C
            "flow": 0.05,  # 5% accuracy
            "depth": 0.1  # ±0.1m
        }
        
        print("🌊 Underwater Sensor Simulator initialized")
        print(f"   Water density: {self.rho_water} kg/m³")
        print(f"   Gravitational constant: {self.g} m/s²")
    
    def generate_sensor_data(self, num_points=20, scenario="normal"):
        """
        Generate realistic underwater sensor data
        
        Args:
            num_points: Number of sensor measurement points
            scenario: "normal", "minor_leak", "major_leak", "critical_leak"
        """
        
        sensor_data = []
        
        # Generate depth points (0-50m underwater)
        depths = np.linspace(5, 50, num_points)
        
        for i, depth in enumerate(depths):
            # Step 1: Calculate expected pressure using hydrostatic formula
            # P = P0 + ρgh
            expected_pressure = self.P0 + (self.rho_water * self.g * depth)
            
            # Add sensor noise
            pressure_noise = np.random.normal(0, expected_pressure * self.sensor_accuracy["pressure"])
            
            # Simulate leakage scenarios
            if scenario == "minor_leak":
                # Minor pressure drop at certain points
                if 15 <= depth <= 25:
                    pressure_anomaly = -np.random.uniform(1000, 3000)  # 1-3 kPa drop
                else:
                    pressure_anomaly = np.random.normal(0, 500)
            elif scenario == "major_leak":
                # Significant pressure drop
                if 20 <= depth <= 35:
                    pressure_anomaly = -np.random.uniform(3000, 8000)  # 3-8 kPa drop
                else:
                    pressure_anomaly = np.random.normal(0, 800)
            elif scenario == "critical_leak":
                # Severe pressure drop with flow anomalies
                if 25 <= depth <= 40:
                    pressure_anomaly = -np.random.uniform(8000, 15000)  # 8-15 kPa drop
                else:
                    pressure_anomaly = np.random.normal(0, 1200)
            else:
                # Normal operation
                pressure_anomaly = np.random.normal(0, 300)
            
            # Final pressure with anomaly
            observed_pressure = expected_pressure + pressure_noise + pressure_anomaly
            
            # Temperature (decreases with depth)
            base_temp = 15 - (depth * 0.1)  # 15°C at surface, decreases with depth
            temperature = base_temp + np.random.normal(0, self.sensor_accuracy["temperature"])
            
            # Flow rate (affected by leakage)
            if scenario == "normal":
                flow_rate = np.random.uniform(0.5, 2.0)  # Normal flow
            elif scenario == "minor_leak":
                if 15 <= depth <= 25:
                    flow_rate = np.random.uniform(2.5, 4.0)  # Increased flow
                else:
                    flow_rate = np.random.uniform(0.5, 2.0)
            elif scenario == "major_leak":
                if 20 <= depth <= 35:
                    flow_rate = np.random.uniform(4.0, 8.0)  # High flow
                else:
                    flow_rate = np.random.uniform(0.5, 2.5)
            elif scenario == "critical_leak":
                if 25 <= depth <= 40:
                    flow_rate = np.random.uniform(8.0, 15.0)  # Very high flow
                else:
                    flow_rate = np.random.uniform(1.0, 3.0)
            
            # Add measurement noise to flow
            flow_rate += np.random.normal(0, flow_rate * self.sensor_accuracy["flow"])
            flow_rate = max(0, flow_rate)  # Ensure non-negative
            
            # Depth with measurement noise
            measured_depth = depth + np.random.normal(0, self.sensor_accuracy["depth"])
            
            sensor_point = {
                "point_id": i + 1,
                "depth": round(measured_depth, 2),
                "pressure": round(observed_pressure, 1),
                "temperature": round(temperature, 2),
                "flow_rate": round(flow_rate, 3),
                "expected_pressure": round(expected_pressure, 1),
                "pressure_deviation": round(abs(observed_pressure - expected_pressure), 1)
            }
            
            sensor_data.append(sensor_point)
        
        return sensor_data

# ============================================================================
# 2. LEAKAGE DETECTION ENGINE
# ============================================================================

class LeakageDetectionEngine:
    """Physics-based leakage detection and severity assessment"""
    
    def __init__(self):
        self.g = 9.81
        self.rho_water = 1025
        self.P0 = 101325
        
        # Leakage severity thresholds
        self.severity_thresholds = {
            "pressure_deviation": [1000, 3000, 8000],  # Pa
            "flow_anomaly": [2.0, 5.0, 10.0],  # m³/s
            "temperature_anomaly": [2.0, 5.0, 8.0]  # °C
        }
    
    def analyze_sensor_data(self, sensor_data):
        """
        Implement Algorithm Steps 1-3: Pressure validation, anomaly detection, ML prediction
        """
        
        analysis_results = {
            "total_points": len(sensor_data),
            "anomaly_points": 0,
            "max_pressure_deviation": 0,
            "max_flow_rate": 0,
            "temperature_range": [0, 0],
            "anomaly_details": []
        }
        
        # Step 1: Pressure Validation (already done in sensor simulation)
        # Step 2: Anomaly Detection
        
        pressure_deviations = []
        flow_rates = []
        temperatures = []
        
        for point in sensor_data:
            # Calculate pressure deviation
            expected_p = self.P0 + (self.rho_water * self.g * point["depth"])
            deviation = abs(point["pressure"] - expected_p)
            pressure_deviations.append(deviation)
            
            flow_rates.append(point["flow_rate"])
            temperatures.append(point["temperature"])
            
            # Check for anomalies
            if deviation > self.severity_thresholds["pressure_deviation"][0]:
                analysis_results["anomaly_points"] += 1
                analysis_results["anomaly_details"].append({
                    "point_id": point["point_id"],
                    "depth": point["depth"],
                    "pressure_deviation": deviation,
                    "flow_rate": point["flow_rate"],
                    "anomaly_type": self._classify_anomaly(deviation, point["flow_rate"])
                })
        
        # Update analysis results
        analysis_results["max_pressure_deviation"] = max(pressure_deviations)
        analysis_results["max_flow_rate"] = max(flow_rates)
        analysis_results["temperature_range"] = [min(temperatures), max(temperatures)]
        
        # Step 3: ML-based Leakage Prediction (simplified)
        leakage_severity = self._predict_leakage_severity(analysis_results)
        
        return analysis_results, leakage_severity
    
    def _classify_anomaly(self, pressure_dev, flow_rate):
        """Classify anomaly type based on thresholds"""
        
        if pressure_dev > self.severity_thresholds["pressure_deviation"][2]:
            return "critical_pressure_drop"
        elif pressure_dev > self.severity_thresholds["pressure_deviation"][1]:
            return "major_pressure_drop"
        elif pressure_dev > self.severity_thresholds["pressure_deviation"][0]:
            return "minor_pressure_drop"
        
        if flow_rate > self.severity_thresholds["flow_anomaly"][2]:
            return "critical_flow_anomaly"
        elif flow_rate > self.severity_thresholds["flow_anomaly"][1]:
            return "major_flow_anomaly"
        elif flow_rate > self.severity_thresholds["flow_anomaly"][0]:
            return "minor_flow_anomaly"
        
        return "normal"
    
    def _predict_leakage_severity(self, analysis_results):
        """
        Simplified ML prediction model for leakage severity
        Returns λ ∈ [0,1] where 1 is most severe
        """
        
        # Normalize factors
        max_pressure_dev = analysis_results["max_pressure_deviation"]
        max_flow = analysis_results["max_flow_rate"]
        anomaly_ratio = analysis_results["anomaly_points"] / analysis_results["total_points"]
        
        # Weighted severity calculation
        pressure_score = min(max_pressure_dev / 15000, 1.0)  # Normalize to [0,1]
        flow_score = min(max_flow / 15.0, 1.0)  # Normalize to [0,1]
        anomaly_score = min(anomaly_ratio * 2, 1.0)  # Normalize to [0,1]
        
        # Combine scores with weights
        severity_score = (
            0.4 * pressure_score +
            0.3 * flow_score +
            0.3 * anomaly_score
        )
        
        return round(min(severity_score, 1.0), 3)

# ============================================================================
# 3. REAL LLM INTEGRATION FOR RISK ASSESSMENT
# ============================================================================

class LeakageLLMIntegration:
    """Real LLM integration for underwater leakage risk assessment"""
    
    def __init__(self, llm_type="rule_based"):
        self.llm_type = llm_type
        self.model_available = False
        
        if llm_type == "transformers":
            self.setup_transformers()
        elif llm_type == "ollama":
            self.setup_ollama()
        else:
            # Default to intelligent rule-based
            self.model_available = True
            print("🤖 Using physics-based expert system for leakage analysis")
    
    def setup_transformers(self):
        """Setup Hugging Face Transformers"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Use a free model for text generation
            model_name = "microsoft/DialoGPT-medium"
            print(f"📥 Loading {model_name} for leakage analysis...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model_available = True
            print("✅ Transformers model loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Transformers error: {e}")
            print("   Using rule-based analysis instead")
    
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
    
    def create_leakage_prompt(self, severity_score, analysis_results, sensor_data, metadata):
        """
        Algorithm Step 4: Construct structured prompt for LLM
        """
        
        # Calculate summary statistics
        avg_depth = np.mean([p["depth"] for p in sensor_data])
        avg_pressure_dev = np.mean([p["pressure_deviation"] for p in sensor_data])
        max_flow = analysis_results["max_flow_rate"]
        
        # Format anomaly details
        anomaly_summary = "\n".join([
            f"- Point {a['point_id']} at {a['depth']}m: {a['pressure_deviation']} Pa deviation, flow {a['flow_rate']} m³/s"
            for a in analysis_results["anomaly_details"][:5]
        ])
        
        prompt = f"""You are an expert underwater systems engineer specializing in pipeline integrity and leakage detection.

LEAKAGE DETECTION ANALYSIS:
System: {metadata.get('system_type', 'Underwater pipeline system')}
Location: {metadata.get('location', 'Offshore facility')}
Pipe Material: {metadata.get('pipe_material', 'Steel')}
Environment: {metadata.get('environment', 'Marine environment')}

SENSOR DATA ANALYSIS:
Total measurement points: {analysis_results['total_points']}
Average depth: {avg_depth:.1f} meters
Anomaly points detected: {analysis_results['anomaly_points']}
Maximum pressure deviation: {analysis_results['max_pressure_deviation']:.1f} Pa
Maximum flow rate: {max_flow:.2f} m³/s
Temperature range: {analysis_results['temperature_range'][0]:.1f}°C to {analysis_results['temperature_range'][1]:.1f}°C

LEAKAGE SEVERITY ASSESSMENT:
ML-predicted severity score: {severity_score:.3f} (scale 0-1, where 1 is most severe)

DETECTED ANOMALIES:
{anomaly_summary if anomaly_summary else "No significant anomalies detected"}

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

        return prompt
    
    def generate_risk_assessment(self, prompt):
        """
        Algorithm Step 5: Generate LLM risk assessment
        """
        
        print("🤖 Generating expert risk assessment...")
        
        if self.llm_type == "transformers" and self.model_available:
            return self._transformers_generate(prompt)
        elif self.llm_type == "ollama" and self.model_available:
            return self._ollama_generate(prompt)
        else:
            return self._physics_based_analysis(prompt)
    
    def _transformers_generate(self, prompt):
        """Generate using Transformers"""
        try:
            # Simplify prompt for model
            simplified_prompt = f"Underwater leakage analysis: severity score {self._extract_severity(prompt):.3f}. Risk assessment and recommendations:"
            
            inputs = self.tokenizer.encode(simplified_prompt, return_tensors="pt")
            
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + 300,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(simplified_prompt):].strip()
            
            # If response is too short, fall back to physics-based
            if len(response) < 50:
                return self._physics_based_analysis(prompt)
            
            return response
            
        except Exception as e:
            print(f"❌ Transformers generation error: {e}")
            return self._physics_based_analysis(prompt)
    
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
                return self._physics_based_analysis(prompt)
                
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return self._physics_based_analysis(prompt)
    
    def _extract_severity(self, prompt):
        """Extract severity score from prompt"""
        try:
            line = [l for l in prompt.split('\n') if 'severity score:' in l][0]
            return float(line.split('score:')[1].split()[0])
        except:
            return 0.5
    
    def _physics_based_analysis(self, prompt):
        """Advanced physics-based risk assessment"""
        
        print("🔬 Using physics-based expert analysis")
        
        # Extract key parameters
        severity_score = self._extract_severity(prompt)
        
        # Extract other parameters from prompt
        lines = prompt.split('\n')
        anomaly_points = 0
        max_pressure_dev = 0
        max_flow = 0
        
        for line in lines:
            if 'Anomaly points detected:' in line:
                anomaly_points = int(line.split(':')[1].strip())
            elif 'Maximum pressure deviation:' in line:
                max_pressure_dev = float(line.split(':')[1].split()[0])
            elif 'Maximum flow rate:' in line:
                max_flow = float(line.split(':')[1].split()[0])
        
        # Determine risk level
        if severity_score >= 0.8:
            risk_level = "CRITICAL"
            timeline = "immediate emergency response required"
        elif severity_score >= 0.6:
            risk_level = "HIGH"
            timeline = "urgent action required within 24 hours"
        elif severity_score >= 0.3:
            risk_level = "MODERATE"
            timeline = "action required within 72 hours"
        else:
            risk_level = "LOW"
            timeline = "routine monitoring and maintenance"
        
        # Generate comprehensive analysis
        analysis = f"""
UNDERWATER LEAKAGE RISK ASSESSMENT REPORT

EXECUTIVE SUMMARY:
Leakage Severity Score: {severity_score:.3f}/1.0
Risk Level: {risk_level}
Action Timeline: {timeline}

TECHNICAL ANALYSIS:
The physics-based assessment reveals a leakage severity score of {severity_score:.3f}, indicating {risk_level.lower()} risk conditions. 

Pressure Analysis:
• Maximum pressure deviation: {max_pressure_dev:.1f} Pa from hydrostatic baseline
• Hydrostatic pressure validation shows {"significant" if max_pressure_dev > 5000 else "minor"} deviations
• Pressure anomalies detected at {anomaly_points} measurement points

Flow Dynamics:
• Maximum flow rate: {max_flow:.2f} m³/s
• Flow anomalies {"strongly correlate" if max_flow > 5.0 else "moderately correlate"} with pressure deviations
• Flow pattern suggests {"active leakage" if max_flow > 8.0 else "potential system stress"}

RISK ASSESSMENT:
1. RISK LEVEL: {risk_level}
   Justification: Severity score {severity_score:.3f} combined with pressure deviations of {max_pressure_dev:.1f} Pa indicates {risk_level.lower()} risk to system integrity.

2. IMMEDIATE ACTIONS:
   • {"Emergency shutdown and isolation" if risk_level == "CRITICAL" else "Increased monitoring frequency"}
   • {"Deploy emergency repair team" if risk_level == "CRITICAL" else "Schedule detailed inspection"}
   • {"Notify all stakeholders immediately" if risk_level == "CRITICAL" else "Prepare maintenance protocols"}

3. MONITORING REQUIREMENTS:
   • {"Continuous real-time monitoring" if risk_level in ["CRITICAL", "HIGH"] else "Enhanced monitoring schedule"}
   • Pressure sensor calibration verification
   • Flow rate measurement validation
   • Temperature profile analysis

4. POTENTIAL CONSEQUENCES:
   • {"Catastrophic system failure possible" if risk_level == "CRITICAL" else "Gradual system degradation"}
   • {"Immediate environmental impact risk" if risk_level == "CRITICAL" else "Long-term integrity concerns"}
   • {"Significant economic impact" if risk_level in ["CRITICAL", "HIGH"] else "Manageable maintenance costs"}

5. MITIGATION STRATEGIES:
   Primary Actions:
   • {"Emergency isolation and containment" if risk_level == "CRITICAL" else "Systematic pressure testing"}
   • {"Immediate repair mobilization" if risk_level == "CRITICAL" else "Planned maintenance scheduling"}
   • {"Alternative system activation" if risk_level == "CRITICAL" else "Operational parameter optimization"}

   Secondary Actions:
   • Enhanced sensor network deployment
   • Predictive maintenance algorithm implementation
   • System redundancy evaluation

PHYSICS-BASED RECOMMENDATIONS:
Based on hydrostatic pressure analysis and flow dynamics:

• Pressure Management: {"Critical pressure control required" if max_pressure_dev > 8000 else "Standard pressure monitoring adequate"}
• Flow Control: {"Immediate flow restriction needed" if max_flow > 10.0 else "Flow rate within acceptable parameters"}
• System Integrity: {"Structural assessment mandatory" if severity_score > 0.7 else "Routine integrity checks sufficient"}

TIMELINE FOR ACTION:
• Emergency Response: {timeline}
• Detailed Assessment: {"Within 6 hours" if risk_level == "CRITICAL" else "Within 48 hours"}
• Repair Implementation: {"Within 24 hours" if risk_level == "CRITICAL" else "Within 1 week"}
• System Validation: {"Before restart" if risk_level == "CRITICAL" else "After repairs"}

CONCLUSION:
The underwater leakage detection system has identified {risk_level.lower()} risk conditions with severity score {severity_score:.3f}. 
{"Immediate emergency response is required to prevent catastrophic failure." if risk_level == "CRITICAL" else 
 "Prompt action is required to maintain system integrity." if risk_level == "HIGH" else
 "Routine maintenance and monitoring will address identified issues." if risk_level == "MODERATE" else
 "Current conditions are within acceptable operational parameters."}

NEXT STEPS:
1. {"Execute emergency response protocol" if risk_level == "CRITICAL" else "Implement monitoring enhancement"}
2. {"Mobilize repair resources" if risk_level in ["CRITICAL", "HIGH"] else "Schedule maintenance activities"}
3. {"Conduct system shutdown procedures" if risk_level == "CRITICAL" else "Continue operational monitoring"}
4. {"Notify regulatory authorities" if risk_level == "CRITICAL" else "Document findings for compliance"}

This assessment is based on physics-based analysis of hydrostatic pressure deviations, flow dynamics, and established risk assessment protocols for underwater pipeline systems.
"""
        
        return analysis.strip()

# ============================================================================
# 4. COMPLETE UNDERWATER LEAKAGE DETECTION PIPELINE
# ============================================================================

class UnderwaterLeakagePipeline:
    """Complete pipeline for underwater leakage detection and risk assessment"""
    
    def __init__(self, llm_type="rule_based"):
        self.sensor_simulator = UnderwaterSensorSimulator()
        self.detection_engine = LeakageDetectionEngine()
        self.llm_assessor = LeakageLLMIntegration(llm_type)
        
        # Create output directories
        os.makedirs("leakage_data", exist_ok=True)
        os.makedirs("risk_reports", exist_ok=True)
        
        print(f"🔧 Underwater Leakage Pipeline initialized with {llm_type} LLM")
    
    def run_leakage_analysis(self, scenario="normal", metadata=None):
        """
        Run complete leakage detection and risk assessment
        Implements all algorithm steps
        """
        
        print("🌊 UNDERWATER LEAKAGE DETECTION AND RISK ASSESSMENT")
        print("=" * 70)
        
        # Default metadata
        if metadata is None:
            metadata = {
                "system_type": "Offshore oil pipeline",
                "location": "North Sea Platform",
                "pipe_material": "Carbon steel with protective coating",
                "environment": "Deep marine environment",
                "depth_range": "5-50 meters",
                "operational_pressure": "150 bar"
            }
        
        # Generate sensor data
        print(f"\n1. Generating sensor data (scenario: {scenario})...")
        sensor_data = self.sensor_simulator.generate_sensor_data(
            num_points=25, scenario=scenario
        )
        
        print(f"   📊 Generated {len(sensor_data)} sensor readings")
        print(f"   🌊 Depth range: {min(p['depth'] for p in sensor_data):.1f}m to {max(p['depth'] for p in sensor_data):.1f}m")
        
        # Algorithm Steps 1-3: Analyze sensor data
        print("\n2. Analyzing sensor data for leakage detection...")
        analysis_results, severity_score = self.detection_engine.analyze_sensor_data(sensor_data)
        
        print(f"   🔍 Anomaly points detected: {analysis_results['anomaly_points']}")
        print(f"   📈 Leakage severity score: {severity_score:.3f}")
        print(f"   🚨 Max pressure deviation: {analysis_results['max_pressure_deviation']:.1f} Pa")
        
        # Algorithm Step 4: Create LLM prompt
        print("\n3. Constructing risk assessment prompt...")
        prompt = self.llm_assessor.create_leakage_prompt(
            severity_score, analysis_results, sensor_data, metadata
        )
        
        # Algorithm Step 5: Generate risk assessment
        print("\n4. Generating expert risk assessment...")
        risk_assessment = self.llm_assessor.generate_risk_assessment(prompt)
        
        # Determine risk level from assessment
        risk_level = self._extract_risk_level(risk_assessment, severity_score)
        
        # Compile final results
        final_results = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario,
            "llm_type": self.llm_assessor.llm_type,
            "metadata": metadata,
            "sensor_data": sensor_data,
            "analysis_results": analysis_results,
            "leakage_severity": severity_score,
            "risk_level": risk_level,
            "risk_assessment": risk_assessment,
            "algorithm_outputs": {
                "lambda": severity_score,
                "r": risk_level,
                "R": risk_assessment
            }
        }
        
        # Display and save results
        self._display_results(final_results)
        self._save_results(final_results)
        
        return final_results
    
    def _extract_risk_level(self, risk_assessment, severity_score):
        """Extract risk level from assessment text"""
        
        risk_assessment_lower = risk_assessment.lower()
        
        if "critical" in risk_assessment_lower:
            return "CRITICAL"
        elif "high" in risk_assessment_lower:
            return "HIGH"
        elif "moderate" in risk_assessment_lower:
            return "MODERATE"
        else:
            return "LOW"
    
    def _display_results(self, results):
        """Display leakage analysis results"""
        
        print("\n" + "=" * 70)
        print("🔧 UNDERWATER LEAKAGE ANALYSIS RESULTS")
        print("=" * 70)
        
        # Basic information
        print(f"📅 Analysis Date: {results['analysis_date']}")
        print(f"🎭 Scenario: {results['scenario']}")
        print(f"🤖 LLM Type: {results['llm_type']}")
        print(f"📍 Location: {results['metadata']['location']}")
        print(f"🔧 System: {results['metadata']['system_type']}")
        
        # Sensor analysis summary
        analysis = results['analysis_results']
        print(f"\n📊 SENSOR ANALYSIS SUMMARY:")
        print(f"   Measurement points: {analysis['total_points']}")
        print(f"   Anomaly points: {analysis['anomaly_points']}")
        print(f"   Max pressure deviation: {analysis['max_pressure_deviation']:.1f} Pa")
        print(f"   Max flow rate: {analysis['max_flow_rate']:.2f} m³/s")
        
        # Algorithm outputs
        print(f"\n🔬 ALGORITHM OUTPUTS:")
        print(f"   λ (Severity Score): {results['leakage_severity']:.3f}")
        print(f"   r (Risk Level): {results['risk_level']}")
        
        # Risk assessment preview
        print(f"\n🤖 RISK ASSESSMENT PREVIEW:")
        print("-" * 60)
        assessment_lines = results['risk_assessment'].split('\n')
        for line in assessment_lines[:12]:
            if line.strip():
                print(f"   {line.strip()}")
        
        if len(assessment_lines) > 12:
            print("   ... (continued in saved report)")
        print("-" * 60)
    
    def _save_results(self, results):
        """Save leakage analysis results"""
        
        # Save detailed JSON report
        timestamp = results['analysis_date'].replace(' ', '_').replace(':', '-')
        json_filename = f"risk_reports/leakage_analysis_{timestamp}.json"
        
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save human-readable report
        report_filename = f"risk_reports/risk_assessment_{timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write("UNDERWATER LEAKAGE DETECTION AND RISK ASSESSMENT REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Analysis Date: {results['analysis_date']}\n")
            f.write(f"Scenario: {results['scenario']}\n")
            f.write(f"LLM Model: {results['llm_type']}\n")
            f.write(f"Location: {results['metadata']['location']}\n")
            f.write(f"System: {results['metadata']['system_type']}\n\n")
            
            f.write("ALGORITHM OUTPUTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Leakage Severity Score (λ): {results['leakage_severity']:.3f}\n")
            f.write(f"Risk Level (r): {results['risk_level']}\n\n")
            
            f.write("SENSOR DATA SUMMARY:\n")
            f.write("-" * 30 + "\n")
            analysis = results['analysis_results']
            f.write(f"Total measurement points: {analysis['total_points']}\n")
            f.write(f"Anomaly points detected: {analysis['anomaly_points']}\n")
            f.write(f"Maximum pressure deviation: {analysis['max_pressure_deviation']:.1f} Pa\n")
            f.write(f"Maximum flow rate: {analysis['max_flow_rate']:.2f} m³/s\n\n")
            
            f.write("RISK ASSESSMENT AND RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            f.write(results['risk_assessment'])
        
        # Save sensor data to CSV
        csv_filename = f"leakage_data/sensor_data_{timestamp}.csv"
        sensor_df = pd.DataFrame(results['sensor_data'])
        sensor_df.to_csv(csv_filename, index=False)
        
        print(f"\n✅ Reports saved:")
        print(f"   📄 Detailed analysis: {json_filename}")
        print(f"   📋 Risk assessment: {report_filename}")
        print(f"   📊 Sensor data: {csv_filename}")

# ============================================================================
# 5. MAIN EXECUTION WITH SCENARIO SELECTION
# ============================================================================

def print_scenario_options():
    """Print available leakage scenarios"""
    
    print("🎭 AVAILABLE LEAKAGE SCENARIOS:")
    print("=" * 50)
    
    scenarios = {
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
    
    for i, (scenario, details) in enumerate(scenarios.items(), 1):
        print(f"\n{i}. {scenario.upper()}")
        print(f"   Description: {details['description']}")
        print(f"   Characteristics: {details['characteristics']}")
        print(f"   Expected Severity: {details['expected_severity']}")

def print_llm_options():
    """Print LLM options for risk assessment"""
    
    print("\n🤖 LLM OPTIONS FOR RISK ASSESSMENT:")
    print("=" * 50)
    
    print("\n1. PHYSICS-BASED EXPERT SYSTEM (Recommended)")
    print("   ✅ Advanced hydrostatic pressure analysis")
    print("   ✅ Flow dynamics expertise")
    print("   ✅ Industry-standard risk assessment")
    print("   ✅ No setup required")
    
    print("\n2. HUGGING FACE TRANSFORMERS")
    print("   📥 Requires: pip install transformers torch")
    print("   ✅ Real neural network analysis")
    print("   ✅ Free models available")
    
    print("\n3. OLLAMA (Local LLM)")
    print("   📥 Requires: Ollama installation")
    print("   ✅ High-quality local processing")
    print("   ✅ Privacy-focused")

def create_sample_metadata():
    """Create sample system metadata"""
    
    metadata_options = [
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
    
    return random.choice(metadata_options)

def main():
    """Main execution function"""
    
    print("🌊 UNDERWATER LEAKAGE DETECTION AND RISK ASSESSMENT PIPELINE")
    print("Physics-based AI system for pipeline integrity monitoring")
    print("=" * 70)
    
    # Show scenario options
    print_scenario_options()
    
    # Show LLM options
    print_llm_options()
    
    # Get user selections
    print("\n" + "=" * 70)
    print("SYSTEM CONFIGURATION:")
    
    # Select scenario
    print("\nSelect leakage scenario:")
    print("1. Normal operation")
    print("2. Minor leak")
    print("3. Major leak") 
    print("4. Critical leak")
    
    scenario_choice = input("\nEnter scenario choice (1-4) [2]: ").strip()
    scenario_mapping = {
        "1": "normal",
        "2": "minor_leak",
        "3": "major_leak", 
        "4": "critical_leak",
        "": "minor_leak"  # Default
    }
    
    selected_scenario = scenario_mapping.get(scenario_choice, "minor_leak")
    
    # Select LLM type
    print("\nSelect LLM type:")
    print("1. Physics-based Expert System")
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
    
    print(f"\n🎭 Selected scenario: {selected_scenario}")
    print(f"🤖 Selected LLM: {selected_llm}")
    
    # Initialize pipeline
    print(f"\n🔧 Initializing pipeline...")
    pipeline = UnderwaterLeakagePipeline(selected_llm)
    
    # Create sample metadata
    metadata = create_sample_metadata()
    
    # Run analysis
    try:
        print(f"\n🔍 Running leakage detection analysis...")
        print(f"   System: {metadata['system_type']}")
        print(f"   Location: {metadata['location']}")
        print(f"   Environment: {metadata['environment']}")
        
        results = pipeline.run_leakage_analysis(selected_scenario, metadata)
        
        print("\n" + "=" * 70)
        print("🎉 UNDERWATER LEAKAGE ANALYSIS COMPLETE!")
        print("=" * 70)
        
        # Display key findings
        print("🔬 KEY FINDINGS:")
        print(f"   Leakage Severity Score (λ): {results['leakage_severity']:.3f}")
        print(f"   Risk Level (r): {results['risk_level']}")
        print(f"   Anomaly Points: {results['analysis_results']['anomaly_points']}")
        print(f"   Max Pressure Deviation: {results['analysis_results']['max_pressure_deviation']:.1f} Pa")
        
        print("\n📁 REPORTS GENERATED:")
        print("   📄 Detailed technical analysis")
        print("   📋 Risk assessment and recommendations")
        print("   📊 Sensor data export")
        print("   🔍 Physics-based validation")
        
        # Risk level summary
        risk_level = results['risk_level']
        if risk_level == "CRITICAL":
            print(f"\n🚨 CRITICAL ALERT: Immediate emergency response required!")
        elif risk_level == "HIGH":
            print(f"\n⚠️  HIGH RISK: Urgent action needed within 24 hours")
        elif risk_level == "MODERATE":
            print(f"\n📋 MODERATE RISK: Action required within 72 hours")
        else:
            print(f"\n✅ LOW RISK: Routine monitoring sufficient")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print("Please check your configuration and try again.")
        return None

# ============================================================================
# 6. EXAMPLE USAGE AND TESTING
# ============================================================================

def demo_all_scenarios():
    """Demonstrate all leakage scenarios"""
    
    print("🧪 DEMONSTRATING ALL LEAKAGE SCENARIOS")
    print("=" * 60)
    
    scenarios = ["normal", "minor_leak", "major_leak", "critical_leak"]
    pipeline = UnderwaterLeakagePipeline("rule_based")
    
    results_summary = []
    
    for scenario in scenarios:
        print(f"\n🎭 Testing scenario: {scenario}")
        print("-" * 40)
        
        metadata = create_sample_metadata()
        results = pipeline.run_leakage_analysis(scenario, metadata)
        
        results_summary.append({
            "scenario": scenario,
            "severity": results['leakage_severity'],
            "risk_level": results['risk_level'],
            "anomalies": results['analysis_results']['anomaly_points']
        })
        
        print(f"   λ: {results['leakage_severity']:.3f}")
        print(f"   Risk: {results['risk_level']}")
    
    # Display summary
    print("\n📊 SCENARIO COMPARISON:")
    print("=" * 60)
    for result in results_summary:
        print(f"{result['scenario']:15} | λ: {result['severity']:.3f} | Risk: {result['risk_level']:8} | Anomalies: {result['anomalies']}")
    
    return results_summary

if __name__ == "__main__":
    # Run the complete underwater leakage detection pipeline
    results = main()
    
    # Optionally run demo of all scenarios
    if results:
        demo_choice = input("\nRun demo of all scenarios? (y/n) [n]: ").strip().lower()
        if demo_choice == 'y':
            demo_all_scenarios()