#!/usr/bin/env python3
"""
Test script to verify if anusvara/halant normalization differences affect WER.
Uses the exact same WER calculation as evaluation.py
"""

import jiwer
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
import string
import re

def normalize_sentence(sentence, lang_code):
    """Exact copy from evaluation.py"""
    factory = IndicNormalizerFactory()
    normalizer = factory.get_normalizer(lang_code)
    normalized_sentence = normalizer.normalize(sentence)
    return normalized_sentence

def process_text(text, lang_code='hi'):
    """Exact processing from evaluation.py process_sample()"""
    text = text.translate(str.maketrans('', '', string.punctuation + "।।'-॥"))
    text = normalize_sentence(text, lang_code)
    text = re.sub(' +', ' ', text)
    return text

# The exact example from predictions
ref_raw = "जिसके लिए उनका देहान्त के बाद तर्पण किया जाता है"
hyp_raw = "जिसके लिए उनका देहांत के बाद तरपण किया जाता है"

print("=" * 60)
print("RAW STRINGS (before normalization)")
print("=" * 60)
print(f"Reference: {ref_raw}")
print(f"Hypothesis: {hyp_raw}")
print(f"\nWER (raw): {jiwer.wer(ref_raw, hyp_raw):.4f}")

print("\n" + "=" * 60)
print("AFTER EVALUATION.PY NORMALIZATION")
print("=" * 60)
ref_processed = process_text(ref_raw)
hyp_processed = process_text(hyp_raw)

print(f"Reference: {ref_processed}")
print(f"Hypothesis: {hyp_processed}")
print(f"\nWER (processed): {jiwer.wer(ref_processed, hyp_processed):.4f}")

# Show character-level differences
print("\n" + "=" * 60)
print("CHARACTER-LEVEL ANALYSIS")
print("=" * 60)

# Focus on the differing words
words_ref = ref_processed.split()
words_hyp = hyp_processed.split()

print("\nWord-by-word comparison:")
for i, (r, h) in enumerate(zip(words_ref, words_hyp)):
    match = "✓" if r == h else "✗"
    print(f"  {i+1}. [{match}] ref='{r}' vs hyp='{h}'")

# Unicode codepoint analysis for differing words
print("\n" + "=" * 60)
print("UNICODE CODEPOINT ANALYSIS (differing words)")
print("=" * 60)

for r, h in zip(words_ref, words_hyp):
    if r != h:
        print(f"\nRef word: {r}")
        print(f"  Codepoints: {[f'U+{ord(c):04X} ({c})' for c in r]}")
        print(f"Hyp word: {h}")
        print(f"  Codepoints: {[f'U+{ord(c):04X} ({c})' for c in h]}")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
final_wer = jiwer.wer(ref_processed, hyp_processed)
if final_wer > 0:
    print(f"YES - These differences DO affect WER!")
    print(f"WER = {final_wer:.4f} ({final_wer*100:.2f}%)")
    print(f"In a 10-word sentence with 2 'wrong' words, this is expected.")
else:
    print("NO - Normalization handles these differences correctly.")
