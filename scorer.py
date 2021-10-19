import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from argparse import ArgumentParser

mlb = MultiLabelBinarizer(classes=list(range(14)))

def one_hot(x):
    res = np.zeros((len(x),14),dtype=int)
    for i, v in enumerate(x):
        res[i,v] = 1
    return res

def main(args):
    gs = pd.read_csv(args.data_dir).label.values
    gs = mlb.fit_transform([tuple([int(ll) for ll in l.split()]) for l in gs])

    preds = []
    for i in range(1,args.num_ensemble+1):
        tmp = pd.read_csv(os.path.join(args.pred_dir,f"probas_ensemble_{i}.csv"))
        preds.append(one_hot(np.argmax(tmp.values,axis=1)))
    
    #print(preds)
    thres = (args.num_ensemble - 1) // 2 + 1

    pred_vote = preds[0].copy()
    for i in range(1,len(preds)):
        pred_vote += preds[i]
    
    #print('*'*10)
    #print(preds)    
    pred_vote = (pred_vote >= thres).astype(int)

    ensemble_ids = []
    precisions, recalls, fscores = [], [], []
    for i in range(1,args.num_ensemble+1):
        ensemble_ids.append(f"ensemble_{i}")
        #print(gs)
        #print(preds[i-1])
        tmp = precision_recall_fscore_support(gs,preds[i-1],average="micro",labels=list(range(1,14)))
        precisions.append(tmp[0]); recalls.append(tmp[1]); fscores.append(tmp[2])
    
    ensemble_ids.append("vote")
    tmp = precision_recall_fscore_support(gs,pred_vote,average="micro",labels=list(range(1,14)))
    precisions.append(tmp[0]); recalls.append(tmp[1]); fscores.append(tmp[2])

    res = pd.DataFrame({"ensemble_id":ensemble_ids,"precision":precisions,"recall":recalls,"F-score":fscores})
    res.to_csv(os.path.join(args.pred_dir,"score.csv"),index=False)

    if args.printout:
        print(res)

if __name__ == "__main__":
    parser = ArgumentParser(description="score the prediction made on the corpus of DrugProt.")
    parser.add_argument("--data_dir",type=str,help="path to the .csv file containing gold standard labels.")
    parser.add_argument("--pred_dir",type=str,help="path to the prediction .csv file.")
    parser.add_argument("--num_ensemble",type=int,help="size of ensemble")
    parser.add_argument("--printout",action="store_true",help="whether print out instant results or not")
    args = parser.parse_args()
    main(args)
