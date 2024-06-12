import torch
import warnings
from tqdm import tqdm
import jiwer

warnings.filterwarnings("ignore")

def testing(test_loader, model, processor ):
    model.eval()

    for batch in tqdm(test_loader):
        if torch.isnan(batch['input_features']).any().item():
            print('------------------------nan value--------------------')
            continue
        with torch.no_grad():
            pred = model(batch['input_features'])
            

        pred_ids = torch.argmax(pred.logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(batch['labels'], group_tokens=False)

        wer = jiwer.wer( label_str, pred_str )

        logging(f" wer is: {wer}\n predicted string is: {pred_str} \n label string is: {label_str} ", "/home/hoosh-2/project/sedava/speech2text/saved" )


def logging(text, dir):
    file_path = dir + "/test_log.txt"

    with open( file_path, 'a' ) as log_file:
        log_file.write( text + ' \n' )
