import torch
from transformers import WhisperProcessor
import torch.nn.functional as F
from datasets import load_dataset
import re

# load WER calculation tools
from evaluate import load

# audio length
# LEN = 1920000

# load the processor, non-compressed
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language='English', task='transcribe')

# define the model class
# we need to do this because we use this wrapper in model compression
# define this class so torch could load model.pt successfully
class What(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None  # Placeholder

    def forward(self, x):
        device = x.device
        generated_ids = self.model.generate(input_features=x.to(device),task="transcribe", language="English")
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription


# Load dataset
dummy = load_dataset('EducativeCS2023/dummy_en_asr')

# evaluation function
def evaluate_model(model, device="cuda"):
    model = model.to(device)
    model.eval()

    results = []
    references = []

    for i in range(len(dummy['train'])):
        # load audio database
        audio = dummy['train'][i]["audio"]["array"]
        audio = torch.from_numpy(audio)

        # convert audio to model inputs
        inputs = processor(audio.cpu(), return_tensors="pt", sampling_rate=16000)
        input_features = inputs.input_features.to(device)

        # prediction
        with torch.no_grad():
            prediction = model(input_features) # audio transcripts ["text1", “text2" ...]
        results.append(prediction)
        references.append(dummy['train'][i]['sentence'].lower())

    # transcriptions clear up
    chars_to_remove_regex = r'[\,\'\.\-\,\۔\!\,\？\:\-\“\%\‘\’\”\،\"]'
    predictions = [re.sub(chars_to_remove_regex, '', str(x).lower()) for x in results]


    wer_metric = load("wer")
    print("\n---------------------------------------\n", results) # use results[0] for the first transcript
    wer_score = wer_metric.compute(predictions=predictions, references=references)

    print(f"Total WER Over Dataset: {wer_score}")
    return wer_score

# Load local model
model_path = "model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load(model_path, map_location=device)
model = model.to(device)

# evaluate
wer = evaluate_model(model, device=device)
# print(f"Final WER: {wer}")

# What.model()
