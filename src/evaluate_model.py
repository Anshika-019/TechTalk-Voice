import csv
from TTS.utils.synthesizer import Synthesizer

# Load models
english_model = Synthesizer(model_path="models/fine_tuned_english_model/fine_tuned_model.pth")
regional_model = Synthesizer(model_path="models/fine_tuned_regional_model/fine_tuned_model.pth")

# Sample sentences
english_sentences = ["Test sentence with API and TTS."]
regional_sentences = ["टेस्ट वाक्य।"]

# MOS scoring function (manual input needed for subjective evaluation)
def mos_score(audio_sample):
    # Hypothetical function to get MOS score from human evaluators
    return 4.5

# Evaluate English
with open("evaluation/MOS_scores.csv", mode="w") as file:
    writer = csv.writer(file)
    writer.writerow(["Sample", "MOS Score"])
    for sentence in english_sentences:
        audio = english_model.tts(sentence)
        writer.writerow([sentence, mos_score(audio)])

# Repeat for regional sentences
