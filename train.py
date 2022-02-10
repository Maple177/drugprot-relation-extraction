from __future__ import absolute_import, division, print_function

import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import transformers 
from transformers import (BertConfig, RobertaConfig)

from tqdm import tqdm, trange
from sklearn.metrics import f1_score
from opt import get_args
from loader import DataLoader
from modeling import (train, evaluate,load_model,set_seed)
from bert_model import BertForSequenceClassification

transformers.logging.set_verbosity_error()
logger = logging.getLogger(__name__)

os.environ["MKL_DEBUG_CPU_TYPE"] = '5'
NUM_NEW_WPS = 29020

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
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %H:%S',level=logging.INFO,filename=f"training_log_{args.run_id}",filemode='w')
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu)
    logger.info(f"learning rate: {args.learning_rate}")

    if not args.config_name_or_path:
        config_file_name = f"./drugprot-relation-extraction/config/{args.bert_variant}.json"
        assert os.path.exists(config_file_name), "requested BERT model variant not in the preset. You can place the corresponding config file under the folder /config/"
        args.config_name_or_path = config_file_name

    config = BertConfig.from_pretrained(args.config_name_or_path,
                                        num_labels=args.num_labels)

    output_dir = os.path.join(args.finetuned_model_path,f"{args.bert_variant}_{args.model_type}",f"bs_{args.batch_size}_lr_{args.learning_rate}_num_extra_layers_{args.num_syntax_layers}",
                                                         "training_record")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dataloader = DataLoader(args,"train")
    dev_dataloader = DataLoader(args,"dev",eval=True)

    all_dev_loss, all_dev_score, all_best_epochs = [], [], []
    # Evaluate the best model on Test set
    for ne in range(args.num_ensemble):
        logger.info(f"<<<<<<<<<<  ensemble_{ne+1}  >>>>>>>>>>>")
        torch.cuda.empty_cache()
        
        set_seed(args,ne)
        model = load_model(args.pretrained_model_path,args.bert_variant,config=config,output_loading_info=False,
                           model_type=args.model_type,num_syntax_layers=args.num_syntax_layers)

        if args.model_type == "with_const_tree":
            model.resize_token_embeddings(NUM_NEW_WPS)
            assert model.bert.embeddings.word_embeddings.weight.shape[0] == NUM_NEW_WPS, "MODEL_TYPE set to with_const_tree, but embedding layer not resized."
        model.to(args.device)

        nb_epochs, loss_record, score_record, best_epoch = train(args,train_dataloader,dev_dataloader,model,ne+1)
        
        all_best_epochs.append(best_epoch)
        all_dev_loss.append(loss_record+[float("NaN")]*(nb_epochs-len(loss_record)))
        all_dev_score.append(score_record+[float("NaN")]*(nb_epochs-len(score_record)))
        
    all_dev_loss = np.array(all_dev_loss)
    all_dev_score = np.array(all_dev_score)
    
    df_loss_record = pd.DataFrame({"members":[f"ensemble_{i}" for i in range(1,args.num_ensemble+1)],**{f"epoch_{i}":all_dev_loss[:,i] for i in range(nb_epochs)}})
    df_loss_record.to_csv(os.path.join(output_dir,"training_record_loss.csv"),index=False)

    df_score_record = pd.DataFrame({"members":[f"ensemble_{i}" for i in range(1,args.num_ensemble+1)],**{f"epoch_{i}":all_dev_score[:,i] for i in range(nb_epochs)}})
    df_score_record.to_csv(os.path.join(output_dir,"training_record_score.csv"),index=False)

    df_best_epochs = pd.DataFrame({"members":[f"ensemble_{i}" for i in range(1,args.num_ensemble+1)],"best_epoch":all_best_epochs})
    df_best_epochs.to_csv(os.path.join(output_dir,"training_best_epochs.csv"),index=False)
    
    end_time = time.time()
    logger.info(f"time consumed (training): {(end_time-start_time):.3f} s.")
    logger.info("training record saved.")
    
if __name__ == "__main__":
    main()
