# export_onnx.py

import torch
import onnx
from model import CNNVoiceCommandModel
import os

def export():
    with open("vocab.txt", "r") as f:
        vocab_size = len([line for line in f.read().splitlines() if line.strip()])
        
    model = CNNVoiceCommandModel(num_classes=vocab_size)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    
    # .eval() locks the model. It automatically disables Dropout features during ONNX export.
    model.eval() 
    
    # Simulates the strict 1D tensor shape of (1, 80000) for 5.0 seconds of raw 16kHz audio.
    dummy_input = torch.randn(1, 1, 80000) 
    
    output_file = "voice_command_model.onnx"
    
    torch.onnx.export(
        model, 
        dummy_input, 
        output_file, 
        input_names=["input"], 
        output_names=["output"],
        opset_version=18
    )
    
    print(f"Model exported to {output_file}.")
    print("Crucial: All Batch Normalization parameters are now mathematically fused into the convolutional weights.")

if __name__ == "__main__":
    export()