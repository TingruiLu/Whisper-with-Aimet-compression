import os
from decimal import Decimal
import torch
# from torchvision import models
import aimet_common.defs
import aimet_torch.defs
import aimet_torch.utils
from aimet_common.utils import start_bokeh_server_session
from aimet_torch.compress import ModelCompressor
from aimet_torch.visualize_serialized_data import VisualizeCompression
import csv
from jiwer import cer
from transformers import AutoProcessor
# import torchaudio
import torch.nn.functional as F
import pickle
import os
from decimal import Decimal
# Compression-related imports
from aimet_common.defs import CostMetric, CompressionScheme, GreedySelectionParameters, RankSelectScheme
from aimet_torch.defs import WeightSvdParameters, SpatialSvdParameters, ChannelPruningParameters, \
    ModuleCompRatioPair
from aimet_torch.compress import ModelCompressor
from datasets import load_dataset
import torch
import pandas as pd
import re
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate

from torch import nn

from bokeh.command.bootstrap import main

args = [
    "serve",
    "--allow-websocket-origin=localhost:8088",
    "--port=8088"
]


global score
score = []
# LEN=240000
LEN=160000

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language='English', task='transcribe')

# a adapter to make sure aimet can call the forward feature for generated_ids
class What(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    def forward(self, x):
        device = x.device

        generated_ids = self.model.generate(input_features=x.to(device), task='transcribe', language='English')

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        string = ''.join(transcription)
        return string
    

def model_compression_with_visualization(eval_func, model):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    process = None

    try:
        visualization_url, process = start_bokeh_server_session()

        input_shape = (1, 80, 3000)
        

        # Specify the necessary parameters
        greedy_params = GreedySelectionParameters(target_comp_ratio=0.8,
                                                num_comp_ratio_candidates=6,)
                                                # saved_eval_scores_dict='./data/model_eval_scores.pkl')
        rank_select = RankSelectScheme.greedy
        auto_params = WeightSvdParameters.AutoModeParams(rank_select_scheme=rank_select,
                                                        select_params=greedy_params,
                                                        modules_to_ignore=[])

        params = WeightSvdParameters(mode=WeightSvdParameters.Mode.auto,
                                    params=auto_params)

        # Single call to compress the model
        results = ModelCompressor.compress_model(model,
                                                eval_callback=eval_func,
                                                eval_iterations=6,
                                                input_shape=input_shape,
                                                compress_scheme=CompressionScheme.weight_svd,
                                                cost_metric=CostMetric.mac,
                                                parameters=params,)
                                                # visualization_url='http://localhost:8088/')

        compressed_model, stats = results
        print(compressed_model)
        print(stats)     # Stats object can be pretty-printed easily
        torch.save(model,'model.pt')

        comp_ratios_file_path = './data/greedy_selection_comp_ratios_list.pkl'
        eval_scores_path = './data/greedy_selection_eval_scores_dict.pkl'

        # A user can visualize the eval scores dictionary and optimal compression ratios by executing the following code.
        """
        Due to some updates of aimet and bokeh, this part is used only for saving the pkl files.
        It will throw some errors in the end, but will not affect the model compression
        Graphs will be plot manually in bokeh_visualize.py
        """
        compression_visualizations = VisualizeCompression(visualization_url)
        compression_visualizations.display_eval_scores(eval_scores_path)
        compression_visualizations.display_comp_ratio_plot(comp_ratios_file_path)
    finally:
        if process:
            process.terminate()
            process.join()

dummy = load_dataset('EducativeCS2023/dummy_en_asr')

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language='English', task='transcribe')

def evaluate_model(model, iteration, use_cuda):
    # device = 'cuda' if use_cuda else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    results = []
    references=[]

    # Load the audio from dataset
    for i in range(len(dummy['train'])):
        audio = dummy['train'][i]["audio"]["array"]
        audio=torch.from_numpy(audio)

        # Aimet needs a fixed input, so we drop the audio that is too long, and pad the audio that is too short
        if audio.size(0) > LEN : 
            continue
        audio = F.pad(audio,(0,LEN-audio.size(0)))
        # convert the audio to model inputs
        inputs = processor(audio.cpu(), return_tensors="pt", sampling_rate=16000)
        input_features = inputs.input_features
        string = model(input_features.to(device))
        results.append(string)
        references.append(dummy['train'][i]['sentence'].lower())

    chars_to_remove_regex = '[\,\'\.\-\,\۔\!\,\؟\:\-\“\%\‘\’\”\،\"]'
    predictions = [re.sub(chars_to_remove_regex, '', str(x).lower()) for x in results]


    # wer_metric = load_metric("wer")
    wer_metric = evaluate.load("wer")

    wer_score = wer_metric.compute(predictions=predictions, references=references)
    #print("wer score", wer_score)
    
    score.append(float(len(predictions) - wer_score))
    return float(len(predictions) - wer_score)

print(evaluate_model(What().cuda(),1,True))
import pickle

# save the scores
# Not actually used, since Aimet will print out the summary automatically
with open('score_1.pkl', 'wb') as f:
    pickle.dump(score, f)

# print(score)
model_compression_with_visualization(evaluate_model, What().cuda())
