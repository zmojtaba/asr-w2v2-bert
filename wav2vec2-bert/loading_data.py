import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import  Dataset, random_split, DataLoader
import os
import re
from collections import Counter


class AudioDataset(Dataset):
    def __init__(self, file_path, processor, device):
        self.file_path          = file_path
        self.csv_file           = file_path + "/subtitle.csv"
        self.df                 = pd.read_csv( self.csv_file)
        self.processor          = processor 
        self.device             = device
    
    def __len__(self):
        return len( self.df )
        # return int(100)

    def normalize( self, wave ):
        max_value = torch.max( torch.abs( wave ) )
        return wave / max_value
    
    def load_audio(self, audio_path):
        waveform, sample_rate   = torchaudio.load(audio_path)
        resampl_fn              = torchaudio.transforms.Resample(sample_rate, 16000)

        # for stereo audio
        if len(waveform) > 1:
            speech = torch.mean( resampl_fn(waveform), dim=0, keepdim=True )
            return speech, sample_rate
        speech                  = resampl_fn(waveform)
        return speech, sample_rate
    

    
    def __getitem__(self, index):
        audio_path      = os.path.join(self.file_path + "/chunked_audio/", self.df.iloc[index]['file_name'] + ".mp3" )
        waveform, sample_rate        = self.load_audio(audio_path)
        input_features    = self.processor( waveform, return_tensors='pt', padding=True, sampling_rate =16000 )['input_features'][0].to(self.device)
        sentence        = self.df.iloc[index]['subtitle']
        sentence        = re_preprocessing(sentence)

        return input_features.squeeze(), sentence

def re_preprocessing( texts_list ):
    def regex_oprator(text):

        text = re.sub(r'[a-zA-Z0-9]', '', text)
        persian_pattern = re.compile(r'[^\u0600-\u06FF\u0750-\u077F\s]')
        text = re.sub(persian_pattern, '', text)
        chars_to_remove = ['؛', 'ِ', 'ـ', 'ْ', 'ٌ', 'َ', 'ّ', '٬','ٔ','ً', 'ُ', '؟', '،',  'ٓ', 'ٰ', 'ٍ', '٪', '.', '\xa0', '\t' ]

        pattern = '[' + re.escape(''.join(chars_to_remove)) + ']'
        text = re.sub(pattern, '', text)


        text = re.sub( '\u200c', '', text )

        # Remove multiple space chars
        # text = "".join(text.split()).strip()
        return text
    if np.isscalar(texts_list):
        return regex_oprator(texts_list)
    
    preprocessed_texts = []
    for text in texts_list:
        text = regex_oprator(text)
        preprocessed_texts.append(text)

    return preprocessed_texts


def create_vocab( text_list ):
    all_text = "".join(text_list)
    vocab_counter= Counter(all_text)
    vocab = {char: idx for idx, char in enumerate(vocab_counter.keys())}
    return vocab


def spliting_dataset(dataset, split_size):
    """
    split_size is the size of test, meaning precetnage of the dataset that belong to test dataset
    """
    test_size                   = int( split_size * len(dataset) )
    train_size                  = int(len(dataset) - test_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size ] )
    return train_dataset, test_dataset


def data_collator(batch, processor, device):
    input_features    = [item[0] for item in batch]
    sentences       = [item[1] for item in batch]

    input_features    = processor.pad({"input_features": input_features}, padding=True, return_tensors="pt").input_features

    labels          = processor.tokenizer(sentences, padding=True, return_tensors="pt")
    labels = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)

    return {"input_features": input_features.to(device), "labels": labels.to(device)}


def load_df(path):
    return pd.read_csv(path+ "/subtitle.csv")

def data_loader(dataset, batch_size, processor, device ):
    loader = DataLoader( dataset, batch_size=batch_size, collate_fn=lambda batch: data_collator(batch, processor, device), shuffle=True )
    return loader