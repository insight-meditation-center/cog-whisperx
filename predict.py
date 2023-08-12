# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
os.environ['HF_HOME'] = '/src/hf_models'
os.environ['TORCH_HOME'] = '/src/torch_models'
from cog import BasePredictor, Input, Path
import torch
import whisperx
import json


compute_type="float16"
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        self.model = whisperx.load_model("large-v2", self.device, language="en", compute_type=compute_type)

    @torch.inference_mode()
    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        batch_size: int = Input(description="Parallelization of input audio transcription", default=32),
    ) -> str:
        """Run a single prediction on the model"""
        result = self.model.transcribe(str(audio), batch_size=batch_size) 
        # result is dict w/keys ['segments', 'language']
        # segments is a list of dicts,each dict has {'text': <text>, 'start': <start_time_msec>, 'end': <end_time_msec> }
        return ''.join([val['text'] for val in result['segments']])

