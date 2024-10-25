import torch
from torch.quantization import quantize_dynamic

# Load model
model = torch.load("models/fine_tuned_english_model/fine_tuned_model.pth")
quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Save quantized model
torch.save(quantized_model.state_dict(), "optimization/quantized_model.pth")

# Log inference speed
with open("optimization/inference_times.txt", "w") as f:
    # Test inference time here and record
    f.write("Inference time before quantization: ... \n")
    f.write("Inference time after quantization: ... \n")
