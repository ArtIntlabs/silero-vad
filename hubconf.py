dependencies = ['torch', 'torchaudio']
import torch
import json
from utils_vad import (init_jit_model,
                       get_speech_ts,
                       get_speech_ts_adaptive,
                       get_number_ts,
                       get_language,
                       get_language_and_group,
                       save_audio,
                       read_audio,
                       state_generator,
                       single_audio_stream,
                       collect_chunks,
                       drop_chunks)

CACHE_DIR = 'ArtIntlabs_silero-vad_ail-legacy'

def silero_vad(**kwargs):
    """Silero Voice Activity Detector
    Returns a model with a set of utils
    Please see https://github.com/snakers4/silero-vad for usage examples
    """
    hub_dir = torch.hub.get_dir()
    model = init_jit_model(model_path=f'{hub_dir}/{CACHE_DIR}/files/model.jit')
    utils = (get_speech_ts, save_audio, read_audio, collect_chunks)
    return model, utils
