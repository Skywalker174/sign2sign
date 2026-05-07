import torch
import json
import os
import sys

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_PATH)
from evaluate import SignModel 

def export_language(lang):
    model_path = os.path.join(BASE_PATH, 'models', f'{lang}_model.pth')
    scaler_path = os.path.join(BASE_PATH, 'models', f'scaler_{lang}.pth')
    
    if os.path.exists(model_path):
        model = SignModel()
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()
        dummy_input = torch.randn(1, 63)
        onnx_out = os.path.join(BASE_PATH, f"{lang}_model.onnx")
        torch.onnx.export(model, dummy_input, onnx_out, opset_version=11)
        print(f"ASL ONNX Success: {onnx_out}")
    
    if os.path.exists(scaler_path):
        scaler = torch.load(scaler_path, weights_only=False)
        scaler_data = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist()
        }
        json_out = os.path.join(BASE_PATH, f"scaler_{lang}.json")
        with open(json_out, "w") as f:
            json.dump(scaler_data, f)
        print(f"Scaler JSON Success: {json_out}")

if __name__ == "__main__":
    for language in ['asl', 'csl']:
        export_language(language)