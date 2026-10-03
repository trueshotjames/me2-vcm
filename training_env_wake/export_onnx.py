# training_env_wake/export_onnx.py

import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from model import WakeWordModel
import os

def export():
    model = WakeWordModel()
    model.load_state_dict(torch.load("wake_word_model.pth", map_location="cpu"))
    model.eval()
    
    # 1.5 seconds of 16kHz audio representing the 24,000 raw soundwave inputs
    dummy_input = torch.randn(1, 24000) 
    base_onnx_path = "wake_word_model_base.onnx"
    quantized_onnx_path = "wake_word_model_quantized.onnx"

    # 1. Package the AI and embed the log-mel feature extractor into a single file.
    # NOTE: torch >=2.1 uses the dynamo exporter by default; the opset_version kwarg is
    # forwarded but the dynamo exporter may still emit a higher opset (here 18 on torch 2.14).
    torch.onnx.export(
        model,
        dummy_input,
        base_onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18
    )

    # 1b. Strip stale graph.value_info shape annotations left by the dynamo exporter.
    # onnxruntime.quantization.quantize_dynamic calls shape_inference, which aborts with
    # "Inferred shape and existing shape differ in dimension 0: (128) vs (1)" if any
    # value_info entry contradicts the actual graph. Clearing value_info is the standard
    # workaround; shape_inference will repopulate it from the graph itself.
    base_model = onnx.load(base_onnx_path)
    del base_model.graph.value_info[:]
    onnx.save(base_model, base_onnx_path)

    # 2. int8 Dynamic Quantization: Rounds the 32-bit math into smaller 8-bit whole 
    # numbers to shrink the file size by 4x and speed up processing time.
    quantize_dynamic(
        model_input=base_onnx_path,
        model_output=quantized_onnx_path,
        weight_type=QuantType.QUInt8
    )
    
    os.remove(base_onnx_path)

if __name__ == "__main__":
    export()