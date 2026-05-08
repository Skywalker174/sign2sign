import joblib
import json

def scaler_to_json(scaler_path, output_path):
    scaler = joblib.load(scaler_path)
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    with open(output_path, 'w') as f:
        json.dump(scaler_data, f)
    print(f"Scaler exported to {output_path}")

scaler_to_json('models/scaler_asl.pkl', '../WebSign2Sign/src/data/asl_scaler.json')
scaler_to_json('models/scaler_csl.pkl', '../WebSign2Sign/src/data/csl_scaler.json')