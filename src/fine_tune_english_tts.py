import torch
from coqpit import Coqpit
from TTS.config import load_config
from TTS.utils.audio import AudioProcessor
from TTS.utils.synthesizer import Synthesizer
# from TTS.utils.checkpoint_utils import load_checkpoint 

# Model and configuration
config_path = "models/pretrained_english_model/config.json"
checkpoint_path = "models/pretrained_english_model/best_model.pth"
output_path = "models/fine_tuned_english_model/"

# Load configuration
config = load_config(config_path)
audio_processor = AudioProcessor(**config.audio)
model = Synthesizer(model_path=checkpoint_path, config_path=config_path)


# Manually load checkpoint
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])

# Hyperparameters for fine-tuning
learning_rate = 0.0001
epochs = 20
batch_size = 16

# Fine-tuning loop
for epoch in range(epochs):
    # Load dataset and fine-tune
    for batch in dataset_loader(batch_size, "data/technical_english_dataset.csv"):
        model.train()
        loss = model(batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item()}")

# Save fine-tuned model
torch.save(model.state_dict(), f"{output_path}fine_tuned_model.pth")
