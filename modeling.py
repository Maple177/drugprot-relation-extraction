from __future__ import absolute_import, division, print_function

import logging
import os
import random
import json

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder

from torch.optim import Adam
from bert_model import BertForSequenceClassification

from tqdm import tqdm, trange
from sklearn.metrics import f1_score
from opt import get_args
from loader import DataLoader


model_download_shortcuts = {"bert":"bert-base-uncased",
                            "biobert":"dmis-lab/biobert-base-cased-v1.1",
                            "scibert":"allenai/scibert_scivocab_uncased"}

soft_max = torch.nn.Softmax(-1)
sigmoid = torch.nn.Sigmoid()
oh = OneHotEncoder(categories=list(range(14)))

logger = logging.getLogger(__name__)

# load models 
def load_model(model_path,bert_variant,**kwargs):
    path = os.path.join(model_path,"model")
    if os.path.exists(model_path):
        bert_model = BertForSequenceClassification.from_pretrained(path,**kwargs)
    else:
        bert_model = BertForSequenceClassification.from_pretrained(model_download_shortcuts[bert_variant],**kwargs)
        bert_model.save_pretrained(path)
    return bert_model

def vote(args,preds,id2label):
    logger.info("making final prediction: voting...")
    rng = np.random.RandomState(args.seed)
    vote_preds = []
    if preds.shape[0] == 1:
        return [id2label[i] for i in preds]
    else:
        for nb in range(preds.shape[1]):
            count = np.bincount(preds[:,nb])
            if (count == count.max()).sum() == 1:
                vote_preds.append(np.argmax(count))
            else:
                possible_tags = np.where(count==count.max())[0]
                vote_preds.append(possible_tags[rng.randint(len(possible_tags))])
    return vote_preds

def one_hot(x):
    res = np.zeros((len(x),14),dtype=np.int)
    for i, v in enumerate(x):
        res[i,v] = 1
    return res

def set_seed(args,ensemble_id):
    seed = args.seed + ensemble_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def evaluate(dataloader,model,inference=False):
    eval_loss = 0.0
    nb_eval_steps = 0
    full_preds = []
    full_golds = [] # gold standard
    model.eval()

    for batch in dataloader:
        with torch.no_grad():
            if inference:
                logits = model(**batch)[0]
            else:
                loss, logits = model(**batch)[:2]

            if not inference:
               preds = logits.detach().cpu().numpy()
               preds = one_hot(np.argmax(preds,axis=1))
               eval_loss += loss.item()
               nb_eval_steps += 1
               full_golds.append(batch["labels"].detach().cpu().numpy().astype(np.int))
            else:
               preds = soft_max(logits)
               preds = preds.detach().cpu().numpy()
            full_preds.append(preds)
    
    full_preds = np.concatenate(full_preds)
    
    if not inference:
        eval_loss = eval_loss / nb_eval_steps
        full_golds = np.concatenate(full_golds)
        return eval_loss, f1_score(full_golds,full_preds,average="micro",labels=list(range(1,14)))
    else:
        return full_preds
    
def train(args, train_dataloader,dev_dataloader,model,ensemble_id):
    """ Train the model """
    if not args.early_stopping:
        NUM_EPOCHS = args.num_train_epochs
    else:
        logger.info(f"early stopping chosen. MAXIMUM number of epochs set to {args.max_num_epochs}.")
        NUM_EPOCHS = args.max_num_epochs

    n_params = sum([p.nelement() for p in model.parameters()])
    logger.info(f'===number of parameters: {n_params}')

    t_total = len(train_dataloader) * NUM_EPOCHS
    
    #print(f"{len(train_dataloader)},{args.train_batch_size}")
    logger.info(f"===number of epochs:{NUM_EPOCHS}; number of steps:{t_total}")
    optimizer = Adam(model.parameters(),lr=args.learning_rate)

    best_model_dir = f"{args.model_type}/ensemble_{ensemble_id}"

    # Train
    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(train_dataloader)}")
    logger.info(f"Num Epochs = {NUM_EPOCHS}")
    logger.info(f"Batch size = {args.batch_size}")

    global_step = 0
    logging_loss, min_loss, prev_dev_loss = 0.0, np.inf, np.inf
    max_score, prev_dev_score = -np.inf, -np.inf
    training_hist = []
    model.zero_grad()
    train_iterator = range(int(NUM_EPOCHS))
    #set_seed(args)  # for reproducibility 

    dev_loss_record = []
    dev_score_record = []
    for epoch in train_iterator:
        tr_loss = 0.0
        logging_loss = 0.0
        grad_norm = 0.0
        #epoch_iterator = tqdm(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            model.train()
            loss = model(**batch)[0]

            loss.backward() # gradient will be stored in the network
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)

            grad_norm += gnorm
                                                
            tr_loss += loss.item()

            optimizer.step()
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and (step + 1) % args.logging_steps == 0:
                # Log metrics
                logger.info(f"training loss = {(tr_loss - logging_loss)/args.logging_steps} | global step = {global_step}")
                logging_loss = tr_loss

        dev_loss, dev_score = evaluate(dev_dataloader,model)
        dev_loss_record.append(dev_loss)
        dev_score_record.append(dev_score)

        logger.info(f"validation loss = {dev_loss} | validation F1-score = {dev_score} | ensemble_id = {ensemble_id} epoch = {epoch}")

        if args.monitor == "loss" and dev_loss < min_loss:
            min_loss = dev_loss
            best_epoch = epoch
            
            # save model
            output_dir = os.path.join(args.finetuned_model_path,best_model_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("new best model! saved.")
        
        if args.monitor == "score" and dev_score > max_score:
            max_score = dev_score
            best_epoch = epoch

            # save model
            output_dir = os.path.join(args.finetuned_model_path,best_model_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)
            torch.save(args,os.path.join(output_dir,"training_args.bin"))
            logger.info("new best model! saved.")
        
        if args.early_stopping and args.monitor == "loss":
            if dev_loss < prev_dev_loss:
                training_hist.append(True)
            else:
                training_hist.append(False)
                if len(training_hist) > args.patience and not np.any(training_hist[-args.patience:]):
                    logger.info(f"early stopping triggered: best loss on validation set: {min_loss} at epoch {best_epoch}.")
                    #train_iterator.close()
                    break
            prev_dev_loss = dev_loss

        if args.early_stopping and args.monitor == "score":
            if dev_score >= prev_dev_score:
                training_hist.append(True)
            else:
                training_hist.append(False)
                if len(training_hist) > args.patience and not np.any(training_hist[-args.patience:]):
                    logger.info(f"early stopping triggered: best F-score on validation set: {max_score} at {best_epoch}.")
                    #train_iterator.close()
                    break
            prev_dev_score = dev_score

        if epoch + 1 == NUM_EPOCHS:
            break

    return NUM_EPOCHS, dev_loss_record, dev_score_record, best_epoch

