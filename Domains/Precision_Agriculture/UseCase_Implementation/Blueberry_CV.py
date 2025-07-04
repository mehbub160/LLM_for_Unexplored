# Simple CV Model Dummy Output Generator
# Just run this to get sensible blueberry ripeness data

import json
from datetime import datetime
import random

def generate_cv_output():
    """Generate realistic blueberry CV analysis output"""
    
    # Realistic farm-scale numbers
    images_taken = random.randint(45, 85)  # 45-85 images per field analysis
    total_berries_detected = random.randint(15000, 35000)  # 15K-35K berries detected
    field_coverage = random.randint(88, 96)  # 88-96% field coverage
    
    # Sensible mid-season blueberry ripeness percentages
    # (These add up to 100%)
    cv_result = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "ripeness_distribution": [12.5, 23.0, 34.5, 26.5, 3.5],  # R1, R2, R3, R4, R5
        "images_processed": images_taken,
        "total_berries_detected": total_berries_detected,
        "berries_per_image": round(total_berries_detected / images_taken),
        "confidence_score": round(random.uniform(0.87, 0.94), 2),
        "field_coverage": f"{field_coverage}%"
    }
    
    return cv_result

# Generate the output
cv_output = generate_cv_output()

# Print it nicely
print("🫐 CV MODEL OUTPUT (Realistic Scale)")
print("=" * 50)
print(f"Date: {cv_output['analysis_date']}")
print(f"Images analyzed: {cv_output['images_processed']}")
print(f"Total berries detected: {cv_output['total_berries_detected']:,}")
print(f"Berries per image: {cv_output['berries_per_image']}")
print(f"Field coverage: {cv_output['field_coverage']}")
print(f"Detection confidence: {cv_output['confidence_score']}")
print(f"")
print(f"Ripeness breakdown:")
print(f"  R1 (Unripe): {cv_output['ripeness_distribution'][0]}% = {int(cv_output['total_berries_detected'] * cv_output['ripeness_distribution'][0] / 100):,} berries")
print(f"  R2 (Early): {cv_output['ripeness_distribution'][1]}% = {int(cv_output['total_berries_detected'] * cv_output['ripeness_distribution'][1] / 100):,} berries")
print(f"  R3 (Developing): {cv_output['ripeness_distribution'][2]}% = {int(cv_output['total_berries_detected'] * cv_output['ripeness_distribution'][2] / 100):,} berries")
print(f"  R4 (Ready): {cv_output['ripeness_distribution'][3]}% = {int(cv_output['total_berries_detected'] * cv_output['ripeness_distribution'][3] / 100):,} berries")
print(f"  R5 (Overripe): {cv_output['ripeness_distribution'][4]}% = {int(cv_output['total_berries_detected'] * cv_output['ripeness_distribution'][4] / 100):,} berries")

# Save to file
with open('cv_dummy_output.json', 'w') as f:
    json.dump(cv_output, f, indent=2)

print(f"\n✅ Saved to: cv_dummy_output.json")
print(f"📊 Total percentage: {sum(cv_output['ripeness_distribution'])}%")

# Show the exact format your pipeline needs
print(f"\n🔗 Use this in your pipeline:")
print("cv_results = {")
print(f"    'analysis_date': '{cv_output['analysis_date']}',")
print(f"    'ripeness_distribution': {cv_output['ripeness_distribution']},")
print(f"    'images_processed': {cv_output['images_processed']},")
print(f"    'confidence_score': {cv_output['confidence_score']},")
print(f"    'field_coverage': '{cv_output['field_coverage']}'")
print("}")