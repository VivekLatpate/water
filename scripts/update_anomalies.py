import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# from app.utils.data_loader import load_groundwater_data
# from app.utils.anomaly_calibration import calibrate_anomaly_detector
# from app.models.anomaly import save_anomaly_detector

def main():
    """Train improved anomaly detector with calibration."""
    
    print("🔍 Training improved anomaly detector...")
    
    # Load data
    df_groundwater = load_groundwater_data("dummy")
    print(f"   Loaded {len(df_groundwater):,} records for {df_groundwater['District'].nunique()} districts")
    
    # Calibrate anomaly detector
    print("   Calibrating anomaly detection parameters...")
    calibration_results = calibrate_anomaly_detector(df_groundwater)
    
    if "error" in calibration_results:
        print(f"❌ Calibration failed: {calibration_results['error']}")
        return
    
    detector = calibration_results["detector"]
    contamination = calibration_results["contamination"]
    
    print(f"   Optimal contamination parameter: {contamination}")
    print(f"   Training samples: {calibration_results['training_samples']}")
    
    # Save calibrated detector
    detector_path = "models/anomaly_isolation_forest_calibrated.pkl"
    save_anomaly_detector(detector, detector_path)
    print(f"✅ Saved calibrated anomaly detector to {detector_path}")
    
    # Test the detector
    print("   Testing anomaly detection...")
    from app.models.anomaly import detect_anomalies
    
    anomalies = detect_anomalies(df_groundwater, detector)
    anomaly_count = sum(anomalies.values())
    anomaly_rate = anomaly_count / len(anomalies) * 100
    
    print(f"   Anomalies detected: {anomaly_count}/{len(anomalies)} ({anomaly_rate:.1f}%)")
    
    # Show districts with anomalies
    anomaly_districts = [district for district, is_anomaly in anomalies.items() if is_anomaly]
    if anomaly_districts:
        print(f"   Districts with anomalies: {', '.join(anomaly_districts[:10])}")
        if len(anomaly_districts) > 10:
            print(f"   ... and {len(anomaly_districts) - 10} more")
    
    # Save calibration results
    import json
    with open("data/processed/anomaly_calibration_results.json", "w") as f:
        json.dump({
            "contamination": contamination,
            "training_samples": calibration_results["training_samples"],
            "sensitivity_analysis": calibration_results["sensitivity_analysis"],
            "anomaly_rate": anomaly_rate,
            "anomaly_districts": anomaly_districts,
            "calibration_date": datetime.now().isoformat()
        }, f, indent=2)
    
    print("✅ Anomaly detection calibration completed")

if __name__ == "__main__":
    main()
