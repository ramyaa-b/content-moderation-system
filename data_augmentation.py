"""
Data Augmentation for Hate Speech Detection
Uses Back-Translation and Synonym Replacement to expand datasets.
"""

import pandas as pd
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer
import nltk
from nltk.corpus import wordnet
import random
from tqdm import tqdm
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HateSpeechAugmentor:
    def __init__(self, device=None):
        # Path relative to backend/ (assuming script is run from backend root)
        self.base_dir = Path("./datasets/hate_speech/processed")
        
        # Auto-detect GPU
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initialized Augmentor on device: {self.device}")
        
        # Download NLTK data for synonyms
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')

        # Model cache (lazy loading to save RAM)
        self.models = {}
        self.tokenizers = {}

    def _load_translation_model(self, model_name):
        """Lazy load translation models to save RAM"""
        if model_name not in self.models:
            logger.info(f"Loading translation model: {model_name}...")
            self.tokenizers[model_name] = MarianTokenizer.from_pretrained(model_name)
            self.models[model_name] = MarianMTModel.from_pretrained(model_name).to(self.device)
        return self.tokenizers[model_name], self.models[model_name]

    def back_translate(self, text, source_lang='en'):
        """
        Performs Back Translation:
        English -> Hindi -> English (for English text)
        Hindi -> English -> Hindi (for Hindi text)
        """
        try:
            if source_lang == 'en':
                # En -> Hi
                tokenizer_fwd, model_fwd = self._load_translation_model("Helsinki-NLP/opus-mt-en-hi")
                # Hi -> En
                tokenizer_bwd, model_bwd = self._load_translation_model("Helsinki-NLP/opus-mt-hi-en")
            elif source_lang == 'hi':
                # Hi -> En
                tokenizer_fwd, model_fwd = self._load_translation_model("Helsinki-NLP/opus-mt-hi-en")
                # En -> Hi
                tokenizer_bwd, model_bwd = self._load_translation_model("Helsinki-NLP/opus-mt-en-hi")
            else:
                return text # Unsupported language for back-translation

            # Forward Translation
            inputs = tokenizer_fwd(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            translated = model_fwd.generate(**inputs)
            text_mid = tokenizer_fwd.batch_decode(translated, skip_special_tokens=True)[0]

            # Backward Translation
            inputs_back = tokenizer_bwd(text_mid, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            back_translated = model_bwd.generate(**inputs_back)
            text_final = tokenizer_bwd.batch_decode(back_translated, skip_special_tokens=True)[0]

            return text_final

        except Exception as e:
            logger.warning(f"Back-translation failed: {e}")
            return text

    def synonym_replacement(self, text, n=1):
        """Replaces n random words with synonyms (English only)"""
        words = text.split()
        if len(words) < 3: return text 

        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalnum()]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = []
            for syn in wordnet.synsets(random_word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
            
            if len(synonyms) >= 1:
                synonym = random.choice(list(set(synonyms)))
                if synonym != random_word:
                    idx = [i for i, x in enumerate(words) if x == random_word][0]
                    new_words[idx] = synonym.replace('_', ' ')
                    num_replaced += 1
            
            if num_replaced >= n:
                break
                
        return ' '.join(new_words)

    def augment_dataset(self, input_path, output_path, samples_per_class=500):
        """Augments the training dataset."""
        logger.info(f"Reading dataset from {input_path}")
        if not Path(input_path).exists():
             logger.error(f"Input file not found: {input_path}")
             return

        df = pd.read_csv(input_path)
        augmented_rows = []
        
        logger.info("Starting augmentation...")
        # Sample a subset to augment (for demo speed)
        subset_to_augment = df.sample(n=min(len(df), samples_per_class * 7), random_state=42)
        
        for idx, row in tqdm(subset_to_augment.iterrows(), total=len(subset_to_augment)):
            original_text = row['text']
            label = row['label']
            lang = row['language']
            
            new_text = None
            method = None
            
            if lang == 'hindi':
                new_text = self.back_translate(original_text, source_lang='hi')
                method = 'back_translation'
            elif lang == 'english':
                if random.random() > 0.5:
                    new_text = self.synonym_replacement(original_text)
                    method = 'synonym_replacement'
                else:
                    new_text = self.back_translate(original_text, source_lang='en')
                    method = 'back_translation'
            
            if new_text and new_text != original_text:
                augmented_rows.append({
                    'text': new_text,
                    'label': label,
                    'language': lang,
                    'source': f"aug_{method}"
                })

        aug_df = pd.DataFrame(augmented_rows)
        final_df = pd.concat([df, aug_df], ignore_index=True)
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Original size: {len(df)} | Augmented added: {len(aug_df)}")
        final_df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved augmented dataset to {output_path}")

if __name__ == "__main__":
    # Run from backend root
    augmentor = HateSpeechAugmentor()
    augmentor.augment_dataset(
        input_path="./datasets/hate_speech/processed/train.csv",
        output_path="./datasets/hate_speech/processed/train_augmented.csv",
        samples_per_class=100 # Low number for quick testing
    )