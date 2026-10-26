import os
import sys

def main():
    print("==========================================================")
    print("ONNX Export and INT8 Quantization Script for BGE-M3")
    print("==========================================================")
    
    # Check dependencies
    try:
        import torch
        import transformers
        import optimum
        import onnxruntime
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as e:
        print(f"Error: Missing dependency: {e.name}.")
        print("Please install them using: pip install torch transformers optimum[onnxruntime] onnxruntime-tools")
        sys.exit(1)

    model_name = "BAAI/bge-m3"
    export_dir = "model_onnx"
    
    print(f"1. Loading tokenizer and exporting {model_name} to FP32 ONNX...")
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Export model to FP32 ONNX format
        model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
        
        # Save FP32 ONNX model
        model.save_pretrained(export_dir)
        tokenizer.save_pretrained(export_dir)
        print(f"FP32 ONNX model exported and saved to {export_dir}/")
    except Exception as e:
        print(f"Error during FP32 export: {e}")
        sys.exit(1)

    model_fp32_path = os.path.join(export_dir, "model.onnx")
    model_quantized_path = os.path.join(export_dir, "model_quantized.onnx")
    
    if not os.path.exists(model_fp32_path):
        print(f"Error: model.onnx not found at {model_fp32_path}")
        sys.exit(1)
        
    print(f"2. Performing dynamic INT8 quantization on {model_fp32_path}...")
    try:
        quantize_dynamic(
            model_input=model_fp32_path,
            model_output=model_quantized_path,
            weight_type=QuantType.QInt8
        )
        print("==========================================================")
        print("SUCCESS!")
        print(f"Quantized model saved to: {model_quantized_path}")
        
        # Calculate sizes
        fp32_size = os.path.getsize(model_fp32_path) / (1024 * 1024)
        quant_size = os.path.getsize(model_quantized_path) / (1024 * 1024)
        reduction = (1.0 - (quant_size / fp32_size)) * 100
        
        print(f"FP32 Model Size: {fp32_size:.2f} MB")
        print(f"INT8 Quantized Model Size: {quant_size:.2f} MB")
        print(f"Size Reduction: {reduction:.1f}%")
        print("==========================================================")
        
    except Exception as e:
        print(f"Error during INT8 quantization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
