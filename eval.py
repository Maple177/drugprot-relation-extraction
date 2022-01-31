from __future__ import absolute_import, division, print_function

import logging
import os
import random
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import (BertConfig, RobertaConfig)
from bert_model import (BertForSequenceClassification, RobertaForSequenceClassification)

from tqdm import tqdm, trange
from sklearn.metrics import f1_score
from opt import get_args
from loader import DataLoader
from modeling import (evaluate, read_in, load_tokenizer, set_seed, vote)

logger = logging.getLogger(__name__)
lr_to_str = {1e-5:"1e-05",2e-5:"2e-05",5e-5:"5e-05",1e-4:"0.0001",} # sometimes under different systems 1e-4 is represented in different ways. 

def main():
    start_time = time.time()
    args = get_args()
    
    # Setup CUDA, GPU & distributed training
    if not args.force_cpu and not torch.cuda.is_available():
        logger.info("NO available GPU. STOPPED. If you want to continue without GPU, add --force_cpu")
        return 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %H:%S',level=logging.INFO,filename="prediction_log",filemode='w')
    logger.warning("device: %s, n_gpu: %s",device,args.n_gpu)

    if not args.config_name_or_path:
        config_file_name = f"./config/{args.bert_variant}.json"
        assert os.path.exists(config_file_name), "requested BERT model variant not in the preset. You can place the corresponding config file under the folder /config/"
        args.config_name_or_path = config_file_name

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # prepare model
    config = BertConfig.from_pretrained(args.config_name_or_path,
                                            num_labels=args.num_labels)

    if args.dev:
        dataloader = DataLoader(args,"dev",eval=True,inference=False)
    else:
        dataloader = DataLoader(args,"test",inference=True)
    
    ensemble_preds = {}
    ensemble_scores = {}

    output_dir = os.path.join(args.finetuned_model_path,f"{args.bert_variant}_{args.model_type}", 
                              f"bs_{args.batch_size}_lr_{lr_to_str[args.learning_rate]}_num_extra_layers_{args.num_syntax_layers}")
    # Evaluate the best model on Test set
    for ne in range(1,args.num_ensemble+1):
        torch.cuda.empty_cache()
        
        set_seed(args,ne)

        ensemble_model_path = os.path.join(output_dir,f"ensemble_{ne}")
        assert os.path.exists(ensemble_model_path), "the selected model is untrained or trained incompletely."
        model = BertForSequenceClassification.from_pretrained(ensemble_model_path,config=config,output_loading_info=False,
                                                              model_type=args.model_type,num_syntax_layers=args.num_syntax_layers)
        model.to(args.device) 

        if args.dev:
            _, tmp_score, tmp_pred = evaluate(dataloader,model)
            ensemble_scores[ne] = tmp_score
        else:
            tmp_pred = evaluate(dataloader,model,inference=True)
        ensemble_preds[ne] = tmp_pred

    if args.dev:
        dest_dir = os.path.join(output_dir,"dev")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        df_score = pd.DataFrame(ensemble_scores)
        df_score.to_csv(os.path.join(dest_dir,"dev_scores.csv"),index=False)
        dev_preds = []
        for ne in range(1,args.num_ensemble+1):
            dev_preds.append(np.argmax(ensemble_preds[ne],1))
        dev_preds = np.concatenate(dev_preds)
        np.save(open(os.path.join(dest_dir,
                                  f"preds_{args.bert_variant}_{args.model_type}_{args.batch_size}_{args.learning_rate}_{args.num_syntax_layers}.npy"),"wb"),
                                  dev_preds)
    else:
        dest_dir = os.path.join(output_dir,"test")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for ne in range(1,args.num_ensemble+1):
            probs = ensemble_preds[ne]
            df = pd.DataFrame({l:probs[:,l] for l in list(range(14))})
            df.to_csv(os.path.join(dest_dir,f"probas_ensemble_{ne}.csv"),index=False)
    
    end_time = time.time()
    logger.info(f"time consumed (prediction):{(end_time-start_time):.3f} s.")
    logger.info("prediction done!")

if __name__ == "__main__":
    main()
