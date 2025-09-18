"""
Water budgeting calculations for sustainable irrigation
"""

import pandas as pd
import numpy as np
from typing import Dict, List

def calculate_weekly_water_budget(crop: str, area_hectares: float, irrigation_method: str) -> Dict:
    """Calculate weekly water budget for a crop"""
    
    # Crop water requirements (mm per week)
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
    
    # Irrigation efficiency factors
    efficiency_factors = {
        'drip': 0.9,
        'sprinkler': 0.8,
        'flood': 0.6,
        'furrow': 0.7
    }
    
    # Get crop requirement
    weekly_requirement_mm = crop_requirements.get(crop.lower(), 25)
    
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
        'crop': crop,
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

def calculate_water_stress_level(water_level: float) -> str:
    """Calculate water stress level based on water level"""
    if water_level > 8:
        return "Low"
    elif water_level > 6:
        return "Moderate"
    elif water_level > 4:
        return "High"
    else:
        return "Critical"

def suggest_alternative_crops(current_crop: str, water_availability: float) -> List[str]:
    """Suggest alternative crops based on water availability"""
    # Crop water requirements (mm per week)
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
    
    # Suggest crops that require less water
    suitable_crops = []
    for crop, requirement in crop_requirements.items():
        if requirement <= water_availability and crop != current_crop:
            suitable_crops.append(crop)
    
    return suitable_crops[:3]  # Return top 3 suggestions

def get_crop_water_requirement(crop: str) -> float:
    """Get water requirement for a specific crop"""
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
    
    return crop_requirements.get(crop.lower(), 25)
