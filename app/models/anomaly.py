"""
Anomaly detection for groundwater levels
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.ensemble import IsolationForest

def detect_anomalies(df: pd.DataFrame, district: str) -> Dict:
    """Detect anomalies in groundwater data for a district"""
    try:
        district_data = df[df['District'] == district].sort_values('Date')
        if len(district_data) < 10:
            return {'anomalies': [], 'total_anomalies': 0}
        
        # Simple anomaly detection based on water level
        water_levels = district_data['WaterLevel_m_bgl'].values
        mean_level = np.mean(water_levels)
        std_level = np.std(water_levels)
        
        # Detect outliers (beyond 2 standard deviations)
        anomalies = []
        for idx, row in district_data.iterrows():
            level = row['WaterLevel_m_bgl']
            if abs(level - mean_level) > 2 * std_level:
                anomalies.append({
                    'date': row['Date'].strftime('%Y-%m-%d'),
                    'severity': 'Moderate',
                    'description': 'Unusual water level detected',
                    'water_level': level
                })
        
        return {
            'district': district,
            'anomalies': anomalies[:5],  # Limit to 5 most recent
            'total_anomalies': len(anomalies)
        }
        
    except Exception as e:
        print(f"Error detecting anomalies for {district}: {e}")
        return {'anomalies': [], 'total_anomalies': 0}
