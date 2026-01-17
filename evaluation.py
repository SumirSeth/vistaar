import argparse
import requests
from datasets import load_dataset, Audio
import evaluate
from joblib import Parallel, delayed
from tqdm import tqdm
import json
import librosa
import pandas as pd
from torch.utils.data import Dataset
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import pyarrow as pa
import soundfile as sf
import jiwer
import os
import string
import re
import time
import torch

lang_to_code = {
    'hindi': 'hi',
    'sanskrit': 'sa',
    'bengali': 'bn',
    'tamil': 'ta',
    'telugu': 'te',
    'gujarati': 'gu',
    'kannada': 'kn',
    'malayalam': 'ml',
    'marathi': 'mr',
    'odia': 'or',
    'punjabi': 'pa',
    'urdu': 'ur',
}

# Mapping from 2-letter language codes to ISO 639-3 format with script (for API)
lang_code_to_api = {
    'hi': 'hin_Deva',
    'sa': 'san_Deva',
    'bn': 'ben_Beng',
    'ta': 'tam_Taml',
    'te': 'tel_Telu',
    'gu': 'guj_Gujr',
    'kn': 'kan_Knda',
    'ml': 'mal_Mlym',
    'mr': 'mar_Deva',
    'or': 'ory_Orya',
    'pa': 'pan_Guru',
    'ur': 'urd_Arab',
}

def normalize_sentence(sentence, lang_code):
    '''
    Perform NFC -> NFD normalization for a sentence and a given language
    sentence: string
    lang_code: language code in ISO format
    '''
    factory=IndicNormalizerFactory()
    normalizer=factory.get_normalizer(lang_code)
    normalized_sentence = normalizer.normalize(sentence)
    return normalized_sentence

class eval_dataset(Dataset):
    
    def __init__(self):
        self.audios = []
        self.sents = []
        
    def __len__(self):
        return len(self.audios)

    def __getitem__(self, i):
        return {"raw": self.audios[i]['array'], "sampling_rate":self.audios[i]['sampling_rate'], "reference":self.sents[i], 
                "path": self.audios[i]['path'], "duration": self.audios[i]['duration']}
    
    def fill_data(self, aud, sent):
        self.audios.append(aud)
        self.sents.append(sent)

def get_data(split, manifest_dir):
    js_data = json.loads(split)
    aud = {}
    # Resolve audio path relative to manifest directory
    audio_path = js_data['audio_filepath']
    if not os.path.isabs(audio_path):
        audio_path = os.path.join(manifest_dir, audio_path)
    
    y, sr = sf.read(audio_path)
    aud['path'] = audio_path
    aud['duration'] = js_data['duration']
    aud['array'] = y
    aud['sampling_rate'] = sr
    
    return (aud, js_data['text'])

def call_api(audio_path, api_url="http://localhost:6769/v1/audio/transcriptions", model="omniASR_LLM_Unlimited_7B_v2", language=None):
    """
    Call the external transcription API with an audio file.
    
    Args:
        audio_path: Path to the audio file
        api_url: API endpoint URL
        model: Model name to use
        language: ISO 639-3 language code with script (e.g., 'hin_Deva')
    
    Returns:
        Transcription text from the API
    """
    try:
        with open(audio_path, 'rb') as f:
            files = {'file': f}
            data = {
                'model': model,
                'response_format': 'verbose_json'
            }
            if language:
                data['language'] = language
            
            response = requests.post(api_url, files=files, data=data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            # Extract text from verbose_json response
            if 'text' in result:
                return result['text']
            elif 'result' in result:
                return result['result']
            else:
                return str(result)  # Fallback to string representation
        else:
            print(f"API Error: Status {response.status_code} for {audio_path}")
            print(f"Response: {response.text}")
            return ""
    except Exception as e:
        print(f"Error calling API for {audio_path}: {str(e)}")
        return ""
    

def main(args):
    
    with open(args.manifest_path, 'r') as f:
        data = f.read()
        splits = data.split('\n')
        if splits[-1] == '':
            splits = splits[:-1]
    
    # Get the benchmarks root directory
    # The manifest is at: /home/vistaar/benchmarks/kathbath/hindi/manifest.json
    # We need to go up 2 directories to get: /home/vistaar/benchmarks/
    manifest_path = os.path.abspath(args.manifest_path)
    manifest_dir = os.path.dirname(manifest_path)  # /home/vistaar/benchmarks/kathbath/hindi
    benchmark_root = os.path.dirname(os.path.dirname(manifest_dir))  # /home/vistaar/benchmarks
    
    da = Parallel(n_jobs=128)(delayed(get_data)(split, benchmark_root) for split in tqdm(splits))
    
    dataset = eval_dataset()
    for d in da:
        dataset.fill_data(d[0], d[1])
    
    hypothesis = []
    ground_truth = []
    
    os.makedirs(dir_path + '/' + 'predictions', exist_ok=True)
    
    out_name = args.model_path.rsplit('/',1)[-1] + '_' + args.manifest_name + '_' + 'predictions.json'
    
    open(dir_path + '/' + 'predictions/' + out_name, 'w').close()
    
    st = time.time()
    
    # Call API for each sample in the dataset
    for i in tqdm(range(len(dataset)), total=len(dataset)):
        sample = dataset[i]
        audio_path = sample['path']
        ref = sample['reference']
        
        # Get the API language code (ISO 639-3 format with script)
        api_lang_code = lang_code_to_api.get(args.lang_code[:2], None)
        
        # Call the external API with language parameter
        hyp = call_api(audio_path, api_url=args.api_url, model=args.model_path, language=api_lang_code)
        
        # Text post-processing
        hyp = hyp.translate(str.maketrans('', '', string.punctuation+"।۔'-॥"))
        ref = ref.translate(str.maketrans('', '', string.punctuation+"।۔'-॥"))
        if args.lang_code[:2] != 'ur':
            hyp = normalize_sentence(hyp, args.lang_code[:2])
            ref = normalize_sentence(ref, args.lang_code[:2])
        hyp = re.sub(' +', ' ', hyp)
        ref = re.sub(' +', ' ', ref)
        
        if ref == '':
            ref = '<empty>'
        hypothesis.append(hyp)
        ground_truth.append(ref)
        res = {
            "audio_filepath": audio_path,
            "duration": sample['duration'],
            "text": ref,
            "pred_text": hyp
        }
        with open(dir_path + '/' + 'predictions/' + out_name, 'a') as f:
            json.dump(res, f)
            f.write('\n')
    
    et = time.time()
     
    data = {}
    data['model'] = args.model_path
    data['dataset'] = args.manifest_name
    data['language'] = args.lang_code
    data['cer'] = jiwer.cer(ground_truth, hypothesis)
    data['time'] = (et-st)/60
    data['batch_size'] = args.batch_size
    measures = jiwer.compute_measures(ground_truth, hypothesis)
    data['wer'] = measures['wer']

    print(data)
    
    with open(dir_path + '/' + 'results.csv', 'a') as results_fp:
        print(','.join([str(v) for v in data.values()]), file=results_fp)


if __name__ == "__main__":
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="model name to use with the API (e.g., omniASR_LLM_Unlimited_7B_v2)",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:6769/v1/audio/transcriptions",
        help="API endpoint URL for transcription",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="path to vistaar manifest",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        required=True,
        help="manifest name",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="The device to run the pipeline on. -1 for CPU (default), 0 for the first GPU and so on. (deprecated)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of samples for each batch. (deprecated)",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="current language",
    )
    args = parser.parse_args()
    
    if len(args.language) == 2:
        args.lang_code = args.language.lower()
    else:
        args.lang_code = lang_to_code[args.language.lower()]

    main(args)
