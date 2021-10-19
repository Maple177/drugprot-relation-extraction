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
        config_file_name = f"./config/{args.model_type}.json"
        assert os.path.exists(config_file_name), "requested BERT model variant not in the preset. You can place the corresponding config file under the folder /config/"
        args.config_name_or_path = config_file_name

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # prepare model
    if args.model_type == "roberta":
        config = RobertaConfig.from_pretrained(args.config_name_or_path,
                                               num_labels=args.num_labels)
    else:
        config = BertConfig.from_pretrained(args.config_name_or_path,
                                            num_labels=args.num_labels)

    tokenizer = load_tokenizer(args.pretrained_model_path,args.model_type)

    model_dir = os.path.join(args.finetuned_model_path,args.model_type)
    test_dataloader = read_in(args,tokenizer,inference=True)
    #id2label = json.load(open(os.path.join(args.finetuned_model_path,"id2label.json"),'r'))
    #id2label = {int(k):v for k,v in id2label.items()}
    
    result = defaultdict(list)
    # Evaluate the best model on Test set
    for ne in range(args.num_ensemble):
        torch.cuda.empty_cache()
        
        set_seed(args,ne)

        ensemble_model_path = os.path.join(args.finetuned_model_path,f"{args.model_type}/ensemble_{ne+1}")
        if args.model_type == "roberta":
            model = RobertaForSequenceClassification.from_pretrained(ensemble_model_path,config=config,output_loading_info=False)
        else:
            model = BertForSequenceClassification.from_pretrained(ensemble_model_path,config=config,output_loading_info=False,
                                                                  with_const=args.with_const,num_syntax_layers=args.num_syntax_layers)
        model.to(args.device) 

        pred = evaluate(test_dataloader,model,predict_only=True)
        result[ne+1].append(pred)

    for ne in range(1,args.num_ensemble+1):
        probs = np.concatenate(result[ne])
        df = pd.DataFrame({l:probs[:,l] for l in list(range(14))})
        df.to_csv(os.path.join(args.output_dir,f"probas_ensemble_{ne}.csv"),index=False)
    
    end_time = time.time()
    logger.info(f"time consumed (prediction):{(end_time-start_time):.3f} s.")
    logger.info("prediction done!")

if __name__ == "__main__":
    main()
