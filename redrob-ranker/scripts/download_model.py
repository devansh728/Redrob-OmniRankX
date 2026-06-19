import os
import torch
from transformers import AutoTokenizer, AutoModel
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    model_id = "BAAI/bge-small-en-v1.5"
    out_dir = os.path.join("models", "bge-small-en-v1.5-int8")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(out_dir)

    model = AutoModel.from_pretrained(model_id)
    model.eval()

    inputs = tokenizer("hello world", return_tensors="pt")
    
    onnx_path = os.path.join(out_dir, "model.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
            onnx_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "token_type_ids": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=14
        )

    quantize_dynamic(
        model_input=onnx_path,
        model_output=onnx_path,
        weight_type=QuantType.QUInt8
    )

if __name__ == "__main__":
    main()
