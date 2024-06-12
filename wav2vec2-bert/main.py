import torch
# /automodel and auto model feature_extractor
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from loading_data import AudioDataset, spliting_dataset, data_loader, create_vocab, re_preprocessing, load_df
from model_config import config
from train_model import train
import warnings
import matplotlib.pyplot as plt
import json
from test_model import testing
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel, AutoProcessor, Wav2Vec2BertProcessor,Wav2Vec2BertForCTC


warnings.filterwarnings("ignore")

device                          =  'cuda:1' if torch.cuda.is_available() else 'cpu'
model_name                      = config.model_name
file_path                       = config.file_path 
batch_size                      = config.batch_size
epochs                          = config.epochs
learning_rate                   = config.learning_rate
adam_betas                      = config.adam_betas
adam_eps                        = config.adam_eps
weight_decay                    = config.weight_decay
warmup_steps                    = config.warmup_steps
gradient_accumulation_steps     = config.gradient_accumulation_steps
max_grad_norm                   = config.max_grad_norm

# creating custom vocab for persian
df = load_df(file_path)
text_list = df['subtitle'].tolist()
text_list = re_preprocessing(text_list)
vocab = create_vocab(text_list)
vocab["|"] = vocab[" "]
del vocab[" "]


vocab_file = "./vocab.json"
with open(vocab_file, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=4)

tokenizer           = Wav2Vec2CTCTokenizer(vocab_file=vocab_file, unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
feature_extractor   = AutoFeatureExtractor.from_pretrained( model_name )
processor           = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

model               = Wav2Vec2BertForCTC.from_pretrained(model_name, ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer), ignore_mismatched_sizes=True).to(device)

dataset             = AudioDataset( file_path, processor, device )


train_dataset, test_dataset         = spliting_dataset(dataset, 0.1)
test_dataset, validation_dataset    = spliting_dataset(test_dataset, 0.5)

train_loader        = data_loader( train_dataset, batch_size, processor, device )
test_loader         = data_loader( test_dataset, batch_size, processor, device )
validation_loader    = data_loader( validation_dataset, batch_size, processor, device )


optimizer       = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=adam_betas, eps=adam_eps, weight_decay=weight_decay )


train(train_loader, validation_loader, model, processor, optimizer, epochs, gradient_accumulation_steps, max_grad_norm)
testing( test_loader, model, processor )


plt.show()