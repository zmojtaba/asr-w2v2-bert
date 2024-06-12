import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from accelerate import Accelerator


previous_val_loss = 10000000

def train(train_loader, validation_loader, model, processor, optimizer, epochs, gradient_accumulation_steps, max_grad_norm ):
    

    accelerator = Accelerator(
    gradient_accumulation_steps=gradient_accumulation_steps,
    )

    (model, optimizer, train_loader, validation_loader) = accelerator.prepare(
    model, optimizer, train_loader, validation_loader
    )

    accelerator.init_trackers("tensorboard_results")
    previous_val_loss = 10000000

    train_loss_list = []
    validation_loss_list = []

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        train_loss = 0.0
        model.train()
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}") ):
            if torch.isnan(batch['input_features']).any().item():
                print('------------------------nan value--------------------')
                continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                train_loss += loss.item()
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # loss = loss / gradient_accumulation_steps
            # loss.backward()
            # if (step + 1) % gradient_accumulation_steps == 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            #     optimizer.step()
            #     optimizer.zero_grad()

        #     batch_loss.append( loss.item() )

        # trian_loss.append( np.mean( batch_loss ) )

        model.eval()
        val_loss = 0.0
        with torch.no_grad():

            for batch in tqdm(validation_loader):
                if torch.isnan(batch['input_features']).any().item():
                    print('------------------------nan value--------------------')
                    continue
               
                pred = model(**batch)
                loss = pred.loss
                val_loss += loss.item()
                    
                pred_ids = torch.argmax(pred.logits, dim=-1)
                pred_str = processor.batch_decode(pred_ids)
                label_str = processor.batch_decode(batch['labels'], group_tokens=False)

        
        avg_val_loss = val_loss / len(validation_loader)
        avg_train_loss = train_loss / len( train_loader )
        train_loss_list.append( avg_train_loss )
        validation_loss_list.append( avg_val_loss )

        ### saving model
        path = "/home/hoosh-2/project/sedava/speech2text/saved/" 

        logg = f""" for {epoch}th epoch train loss is { avg_train_loss  } & val loss is { avg_val_loss  } \n 
label string is: {label_str} \n
predicted string is {pred_str} \n
---------------------------------------------------------------
"""
        logging(logg, path)
        print( f" for {epoch}th epoch train loss is { np.mean( avg_train_loss ) } & val loss is { np.mean( avg_val_loss ) } " )

        if avg_val_loss <= previous_val_loss:
            save_model(model, path + str(epochs) + "th_epoch.pt" )
            # tokenizer.save_pretrained("best_model")
    plot_loss(train_loss_list, validation_loss_list, path + "loss.png" )
    save_model(model, path + str(epochs) + "th_epoch.pt" )
    accelerator.end_training()
    return model, train_loss_list, validation_loss_list


def logging(text, dir):
    file_path = dir + "/log.txt"

    with open( file_path, 'a' ) as log_file:
        log_file.write( text + ' \n' )

    

def save_model(model, path):
    torch.save( model.state_dict(), path )

def plot_loss(train, validation, save_path):
    plt.plot( range(len(train)), train, "-2", label=' trian loss' )
    plt.plot( range(len(validation)), validation, "-o", label=' validation loss' )

    plt.ylabel('loss')
    plt.title( 'loss over epochs' )
    plt.legend()
    plt.savefig(save_path)
    plt.show()




