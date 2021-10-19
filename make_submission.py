import os
import json
import numpy as np
import pandas as pd
from argparse import ArgumentParser

def one_hot(x):
    res = np.zeros((len(x),14),dtype=int)
    for i, v in enumerate(x):
        res[i,v] = 1
    return res

def choose(pred):
    max_p = pred.max()
    res = list(np.where(pred==max_p)[0])
    if len(res) == 1:
        return res
    else:
        return [i for i in res if i != 0]

def main(args):
    df_ori = pd.read_csv(args.data_dir)
    ids, arg1s, arg2s = zip(*df_ori[["article_id","arg1","arg2"]].values.astype(str))

    id2label = json.load(open(os.path.join(args.id2label_dir,"id2label.json"),'r'))
    id2label = {int(i):l for i,l in id2label.items()}

    preds = []
    for i in range(1,args.num_ensemble+1):
        tmp = pd.read_csv(os.path.join(args.pred_dir,f"probas_ensemble_{i}.csv"))
        preds.append(one_hot(np.argmax(tmp.values,axis=1)))

    pred_sum = preds[0]
    for i in range(1,len(preds)):
        pred_sum += preds[i]
    
    # VOTING: 1) choose the label with the maximum count; 2) if multiple labels exist with equal counts, keep all labels except "NULL" (no_relation)

    pred_vote = [choose(pred_sum[i]) for i in range(pred_sum.shape[0])]
    labels = [[id2label[l] for l in ls] for ls in pred_vote]

    labels_to_save = [' '.join(l) for l in labels]
    with open(os.path.join(args.pred_dir,"vote_predictions.txt"),'w') as f:
        f.write('\n'.join(labels_to_save))

    submissions = []
    for i, a1, a2, l in zip(ids,arg1s,arg2s,labels):
        for ll in l:
            if ll != "NULL":
                submissions.append('\t'.join([i,ll,f"Arg1:{a1}",f"Arg2:{a2}"]))
    
    with open(args.output_dir,'w') as f:
        f.write('\n'.join(submissions))

if __name__ == "__main__":
    parser = ArgumentParser(description="make submission files for drugprot tasks.")
    parser.add_argument("--data_dir",type=str,help="path to the original .csv file that was used for the prediction.")
    parser.add_argument("--pred_dir",type=str,help="path to predictions of the neural network.")
    parser.add_argument("--id2label_dir",type=str,help="path to the map of id -> relation label.")
    parser.add_argument("--output_dir",type=str,help="path to store the .tsv submission files.")
    parser.add_argument("--num_ensemble",type=int,help="number of models in an ensemble.")
    args = parser.parse_args()
    main(args)



