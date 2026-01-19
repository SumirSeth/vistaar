import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        return {"reference": self.sents[i], "path": self.audios[i]['path'], "duration": self.audios[i]['duration']}
    
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
    aud['path'] = audio_path
    aud['duration'] = js_data['duration']
    
    return (aud, js_data['text'])

def call_api(audio_path, api_url="http://localhost:6769/v1/audio/transcriptions", model="omniASR_LLM_Unlimited_7B_v2", language=None, endpoint_index=None, num_endpoints=1):
    """
    Call the external transcription API with an audio file.
    
    Args:
        audio_path: Path to the audio file
        api_url: API endpoint URL (base URL if endpoint_index specified)
        model: Model name to use
        language: ISO 639-3 language code with script (e.g., 'hin_Deva')
        endpoint_index: Sample index for distributing across endpoints
        num_endpoints: Number of endpoints to distribute across (starting from port 6769)
    
    Returns:
        Transcription text from the API
    """
    # Select endpoint based on index
    if endpoint_index is not None and num_endpoints > 1:
        port = 6769 + (endpoint_index % num_endpoints)
        actual_url = f"http://localhost:{port}/v1/audio/transcriptions"
    else:
        actual_url = api_url
    
    try:
        with open(audio_path, 'rb') as f:
            files = {'file': f}
            data = {
                'model': model,
                'response_format': 'verbose_json'
            }
            if language:
                data['language'] = language
            
            response = requests.post(actual_url, files=files, data=data, timeout=120)
        
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

def load_existing_predictions(predictions_path):
    """
    Load existing predictions file (JSON lines) and return:
    - processed_paths: set of audio filepaths already processed
    - hypothesis: list of existing hypothesis texts
    - ground_truth: list of existing reference texts
    """
    processed_paths = set()
    hypothesis = []
    ground_truth = []

    if not os.path.exists(predictions_path):
        return processed_paths, hypothesis, ground_truth

    with open(predictions_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                audio_path = item.get("audio_filepath")
                pred_text = item.get("pred_text", "")
                ref_text = item.get("text", "")
                if audio_path:
                    processed_paths.add(audio_path)
                hypothesis.append(pred_text)
                ground_truth.append(ref_text)
            except json.JSONDecodeError:
                continue

    return processed_paths, hypothesis, ground_truth
    

def process_sample(sample, args, api_lang_code, sample_index=None):
    audio_path = sample['path']
    ref = sample['reference']

    # Determine endpoint_index if multiple endpoints enabled
    endpoint_index = sample_index if (args.num_endpoints > 1 and sample_index is not None) else None
    # Don't pass language if --guess flag is set (let model auto-detect)
    lang_param = None if args.guess else api_lang_code
    hyp = call_api(audio_path, api_url=args.api_url, model=args.model_path, language=lang_param, endpoint_index=endpoint_index, num_endpoints=args.num_endpoints)

    # Text post-processing
    hyp = hyp.translate(str.maketrans('', '', string.punctuation+"।।'-॥"))
    ref = ref.translate(str.maketrans('', '', string.punctuation+"।।'-॥"))
    if args.lang_code[:2] != 'ur':
        hyp = normalize_sentence(hyp, args.lang_code[:2])
        ref = normalize_sentence(ref, args.lang_code[:2])
    hyp = re.sub(' +', ' ', hyp)
    ref = re.sub(' +', ' ', ref)

    if ref == '':
        ref = '<empty>'

    res = {
        "audio_filepath": audio_path,
        "duration": sample['duration'],
        "text": ref,
        "pred_text": hyp
    }

    return hyp, ref, res

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
    
    da = Parallel(n_jobs=min(32, os.cpu_count() or 32))(delayed(get_data)(split, benchmark_root) for split in tqdm(splits))
    
    dataset = eval_dataset()
    for d in da:
        dataset.fill_data(d[0], d[1])
    
    # Prepare resumable predictions file
    os.makedirs(dir_path + '/' + 'predictions', exist_ok=True)

    out_name = args.model_path.rsplit('/',1)[-1] + '_' + args.manifest_name + '_' + 'predictions.json'
    predictions_path = dir_path + '/' + 'predictions/' + out_name

    # Load existing predictions if present (resume capability)
    processed_paths, hypothesis, ground_truth = load_existing_predictions(predictions_path)
    
    # Ensure file exists for appending
    if not os.path.exists(predictions_path):
        open(predictions_path, 'w').close()
    
    st = time.time()
    
    # Prepare pending samples
    pending_samples = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample['path'] in processed_paths:
            continue
        pending_samples.append(sample)

    api_lang_code = lang_code_to_api.get(args.lang_code[:2], None)

    if args.num_endpoints > 1:
        # Split samples into N groups for parallel endpoint processing
        grouped_samples = [[] for _ in range(args.num_endpoints)]
        for idx, sample in enumerate(pending_samples):
            group_idx = idx % args.num_endpoints
            grouped_samples[group_idx].append((idx, sample))
        
        def process_batch(samples_with_idx):
            """Process a batch of samples sequentially"""
            results = []
            for idx, sample in samples_with_idx:
                hyp, ref, res = process_sample(sample, args, api_lang_code, idx)
                results.append((hyp, ref, res))
            return results
        
        # Process all batches in parallel (one per endpoint)
        with ThreadPoolExecutor(max_workers=args.num_endpoints) as executor:
            futures = [executor.submit(process_batch, group) for group in grouped_samples]
            
            # Collect results from all endpoints
            all_results = []
            for future in tqdm(as_completed(futures), total=args.num_endpoints, desc="Endpoints"):
                batch_results = future.result()
                all_results.extend(batch_results)
            
            # Write all results and build hypothesis/ground_truth
            for hyp, ref, res in all_results:
                hypothesis.append(hyp)
                ground_truth.append(ref)
                with open(predictions_path, 'a') as f:
                    json.dump(res, f)
                    f.write('\n')
    else:
        # Single-endpoint logic with num_workers threads
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_sample, sample, args, api_lang_code, idx): (sample, idx)
                       for idx, sample in enumerate(pending_samples)}
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                hyp, ref, res = future.result()
                hypothesis.append(hyp)
                ground_truth.append(ref)
                with open(predictions_path, 'a') as f:
                    json.dump(res, f)
                    f.write('\n')
    
    et = time.time()
     
    data = {}
    data['model'] = args.model_path
    data['dataset'] = args.manifest_name
    data['language'] = args.lang_code
    data['cer'] = jiwer.cer(ground_truth, hypothesis)
    data['time'] = (et-st)/60
    data['num_workers'] = args.num_workers
    data['wer'] = jiwer.wer(ground_truth, hypothesis)

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
        "--language",
        type=str,
        required=True,
        help="current language",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of parallel API workers when using single endpoint (default: 2)",
    )
    parser.add_argument(
        "--num_endpoints",
        type=int,
        default=8,
        help="Number of parallel endpoints to distribute load across, starting from port 6769 (default: 8). Set to 1 for single endpoint.",
    )
    parser.add_argument(
        "--guess",
        action="store_true",
        help="Let the model auto-detect language (do not provide language parameter to API)",
    )
    
    args = parser.parse_args()
    
    if len(args.language) == 2:
        args.lang_code = args.language.lower()
    else:
        args.lang_code = lang_to_code[args.language.lower()]

    main(args)
