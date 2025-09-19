"""
Water budgeting calculations for sustainable irrigation
"""

import pandas as pd
import numpy as np
from typing import Dict, List

def calculate_weekly_water_budget(water_level: float, rainfall_probability: float, crop_requirement: Dict, area_hectares: float) -> Dict:
    """Calculate weekly water budget for a crop"""
    
    # Extract crop info from crop_requirement dict
    crop_name = crop_requirement.get('crop', 'wheat')
    weekly_requirement_mm = crop_requirement.get('weekly_requirement_mm', 25)
    irrigation_method = crop_requirement.get('irrigation_method', 'drip')
    
    # Irrigation efficiency factors
    efficiency_factors = {
        'drip': 0.9,
        'sprinkler': 0.8,
        'flood': 0.6,
        'furrow': 0.7
    }
    
    # Calculate total water needed
    total_water_needed = weekly_requirement_mm * area_hectares * 10  # Convert to liters
    
    # Apply irrigation efficiency
    efficiency = efficiency_factors.get(irrigation_method.lower(), 0.8)
    actual_water_needed = total_water_needed / efficiency
    
    # Calculate sustainable budget (80% of requirement)
    sustainable_budget_mm = weekly_requirement_mm * 0.8
    
    # Irrigation frequency and duration
    irrigation_frequency_per_week = 3 if irrigation_method.lower() == 'drip' else 2
    irrigation_duration_minutes = 30 if irrigation_method.lower() == 'drip' else 60
    
    return {
        'crop': crop_name,
        'area_hectares': area_hectares,
        'irrigation_method': irrigation_method,
        'weekly_requirement_mm': weekly_requirement_mm,
        'sustainable_budget_mm': sustainable_budget_mm,
        'total_water_needed_liters': actual_water_needed,
        'irrigation_frequency_per_week': irrigation_frequency_per_week,
        'irrigation_duration_minutes': irrigation_duration_minutes,
        'efficiency_factor': efficiency
    }

def get_available_crops() -> List[str]:
    """Get list of available crops"""
    return ['wheat', 'rice', 'maize', 'sugarcane', 'cotton', 'soybean', 'potato', 'tomato', 'onion', 'chili']

def get_irrigation_methods() -> List[str]:
    """Get list of available irrigation methods"""
    return ['drip', 'sprinkler', 'flood', 'furrow']

def calculate_water_stress_level(water_level: float, district: str = None) -> str:
    """Calculate water stress level based on water level"""
    if water_level > 8:
        return "Low"
    elif water_level > 6:
        return "Moderate"
    elif water_level > 4:
        return "High"
    else:
        return "Critical"

def suggest_alternative_crops(district: str, df_crops: pd.DataFrame, stress_level: str) -> List[Dict]:
    """Suggest alternative crops based on water stress level"""
    
    # Get available crops from dataframe
    available_crops = df_crops['Crop'].unique().tolist() if not df_crops.empty else []
    
    # Filter crops based on stress level
    if stress_level == "Critical":
        # Suggest only low water requirement crops
        suitable_crops = [crop for crop in available_crops if crop.lower() in ['onion', 'chili', 'tomato']]
    elif stress_level == "High":
        # Suggest moderate water requirement crops
        suitable_crops = [crop for crop in available_crops if crop.lower() in ['wheat', 'soybean', 'potato', 'onion', 'chili']]
    else:
        # All crops are suitable
        suitable_crops = available_crops
    
    # Return crop suggestions with basic info
    suggestions = []
    for crop in suitable_crops[:3]:  # Top 3 suggestions
        crop_data = df_crops[df_crops['Crop'] == crop].iloc[0] if not df_crops.empty else {}
        suggestions.append({
            'crop': crop,
            'water_requirement': crop_data.get('Water_Requirement_mm', 25),
            'suitability': 'High' if stress_level in ['Low', 'Moderate'] else 'Moderate'
        })
    
    return suggestions

def get_crop_water_requirement(crop: str, df_crops: pd.DataFrame) -> Dict:
    """Get water requirement for a specific crop"""
    
    # Try to get from dataframe first
    if not df_crops.empty:
        crop_data = df_crops[df_crops['Crop'] == crop]
        if not crop_data.empty:
            crop_info = crop_data.iloc[0]
            return {
                'crop': crop,
                'weekly_requirement_mm': crop_info.get('Water_Requirement_mm', 25),
                'irrigation_method': 'drip',  # Default
                'season': crop_info.get('Season', 'All'),
                'growth_period_days': crop_info.get('Growth_Period_Days', 120)
            }
    
    # Fallback to hardcoded values
    crop_requirements = {
        'wheat': 25,
        'rice': 40,
        'maize': 30,
        'sugarcane': 35,
        'cotton': 28,
        'soybean': 22,
        'potato': 20,
        'tomato': 18,
        'onion': 15,
        'chili': 16
    }
    
    return {
        'crop': crop,
        'weekly_requirement_mm': crop_requirements.get(crop.lower(), 25),
        'irrigation_method': 'drip',
        'season': 'All',
        'growth_period_days': 120
    }
