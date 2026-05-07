import torch
import json
import os
import sys
import onnx

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
        temp_onnx = os.path.join(BASE_PATH, f"temp_{lang}.onnx")
        final_onnx = os.path.join(BASE_PATH, f"{lang}_model.onnx")

        # Step 1: Initial export
        torch.onnx.export(
            model, 
            dummy_input, 
            temp_onnx,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )

        # Step 2: Force embedding weights into a single file using ONNX library
        onnx_model = onnx.load(temp_onnx)
        onnx.save_model(
            onnx_model, 
            final_onnx, 
            save_as_external_data=False # This is the magic fix
        )
        
        # Cleanup temporary files
        if os.path.exists(temp_onnx):
            os.remove(temp_onnx)
        
        # Also cleanup the annoying .data file if it was created during Step 1
        data_file = temp_onnx + ".data"
        if os.path.exists(data_file):
            os.remove(data_file)

        print(f"ONNX Success: {final_onnx}")
    
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