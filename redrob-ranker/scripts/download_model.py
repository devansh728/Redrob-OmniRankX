import os
import shutil
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

def main():
    model_id = "BAAI/bge-small-en-v1.5"
    out_dir = os.path.join("models", "bge-small-en-v1.5-int8")
    temp_dir = os.path.join("models", "bge-small-en-v1.5-temp")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(out_dir)

    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    model.save_pretrained(temp_dir)

    quantizer = ORTQuantizer.from_pretrained(temp_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=out_dir, quantization_config=qconfig)

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    orig_model_path = os.path.join(out_dir, "model.onnx")
    if os.path.exists(orig_model_path):
        os.remove(orig_model_path)
    
    quantized_model_path = os.path.join(out_dir, "model_quantized.onnx")
    dest_model_path = os.path.join(out_dir, "model.onnx")
    if os.path.exists(quantized_model_path):
        os.rename(quantized_model_path, dest_model_path)

if __name__ == "__main__":
    main()
