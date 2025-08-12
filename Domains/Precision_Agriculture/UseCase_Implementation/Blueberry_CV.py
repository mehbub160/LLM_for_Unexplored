##########################
# Blueberry Computer Vision Analysis Output Generator
# Generates realistic dummy data for testing the ripeness detection pipeline
##########################

import json
from datetime import datetime
import random

def generate_blueberry_analysis():
    """
    Creates realistic computer vision analysis results for blueberry ripeness detection.
    Returns a dictionary containing detection metrics and ripeness distribution data.
    """
    
    ##########################
    # Set realistic ranges based on typical farm field analysis
    ##########################
    num_images = random.randint(45, 85)
    detected_berries = random.randint(15000, 35000)
    coverage_percent = random.randint(88, 96)
    
    ##########################
    # Define ripeness percentages that reflect mid-season conditions
    # Categories: R1=Unripe, R2=Early, R3=Developing, R4=Ready, R5=Overripe
    ##########################
    ripeness_percentages = [12.5, 23.0, 34.5, 26.5, 3.5]
    
    analysis_results = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "ripeness_distribution": ripeness_percentages,
        "images_processed": num_images,
        "total_berries_detected": detected_berries,
        "berries_per_image": round(detected_berries / num_images),
        "confidence_score": round(random.uniform(0.87, 0.94), 2),
        "field_coverage": f"{coverage_percent}%"
    }
    
    return analysis_results

def display_analysis_results(results):
    """Print formatted analysis results to console"""
    
    print("COMPUTER VISION MODEL OUTPUT - REALISTIC SCALE")
    print("=" * 50)
    print(f"Analysis Date: {results['analysis_date']}")
    print(f"Images Processed: {results['images_processed']}")
    print(f"Total Berries Detected: {results['total_berries_detected']:,}")
    print(f"Average Berries per Image: {results['berries_per_image']}")
    print(f"Field Coverage: {results['field_coverage']}")
    print(f"Detection Confidence: {results['confidence_score']}")
    print("")
    
    ##########################
    # Calculate and display berry counts for each ripeness stage
    ##########################
    ripeness_labels = ["R1 (Unripe)", "R2 (Early Stage)", "R3 (Developing)", 
                      "R4 (Ready for Harvest)", "R5 (Overripe)"]
    
    print("Ripeness Stage Breakdown:")
    total_berries = results['total_berries_detected']
    
    for i, (label, percentage) in enumerate(zip(ripeness_labels, results['ripeness_distribution'])):
        berry_count = int(total_berries * percentage / 100)
        print(f"  {label}: {percentage}% = {berry_count:,} berries")

def save_results_to_file(results, filename='cv_analysis_output.json'):
    """Save analysis results to JSON file"""
    with open(filename, 'w') as file:
        json.dump(results, file, indent=2)
    print(f"\nResults saved to: {filename}")

def show_pipeline_format(results):
    """Display the exact format needed for integration with analysis pipeline"""
    print(f"\nTotal ripeness percentage validation: {sum(results['ripeness_distribution'])}%")
    print("\nPipeline Integration Format:")
    print("cv_analysis_data = {")
    print(f"    'analysis_date': '{results['analysis_date']}',")
    print(f"    'ripeness_distribution': {results['ripeness_distribution']},")
    print(f"    'images_processed': {results['images_processed']},")
    print(f"    'confidence_score': {results['confidence_score']},")
    print(f"    'field_coverage': '{results['field_coverage']}'")
    print("}")

##########################
# Main execution
##########################
if __name__ == "__main__":
    analysis_data = generate_blueberry_analysis()
    display_analysis_results(analysis_data)
    save_results_to_file(analysis_data)
    show_pipeline_format(analysis_data)
