import torch
import json
class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.device                         =  'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name                     = config.get('model_name')
        self.file_path                      = config.get('file_path', '')
        self.batch_size                     = config.get('batch_size')
        self.epochs                         = config.get('epochs')
        self.learning_rate                  = config.get('learning_rate')
        self.adam_betas                     = tuple(config.get('adam_betas'))
        self.adam_eps                       = config.get('adam_eps')
        self.weight_decay                   = config.get('weight_decay')
        self.warmup_steps                   = config.get('warmup_steps')
        self.gradient_accumulation_steps    = config.get("gradient_accumulation_steps")
        self.max_grad_norm                  = config.get("max_grad_norm")


config = Config( "/home/hoosh-2/project/sedava/speech2text/config.json" )

print()