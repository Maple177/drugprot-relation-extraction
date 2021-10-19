import os
import numpy as np
import pandas as pd
import logging
import torch
import benepar
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=list(range(14)))
try:
    parser = benepar.Parser("benepar_en3")
except:
    import nltk
    benepar.download('benepar_en3')
    parser = benepar.Parser("benepar_en3")

logger = logging.getLogger(__name__)

class DataLoader(object):
    def __init__(self,args,df,tokenizer,tag,eval=False,inference=False):
        self.tokenizer = tokenizer
        self.max_len = args.max_seq_length
        self.inference = inference
        self.device = args.device
        self.add_const = args.with_const
        
        if args.normalise == "target":
            data = list(df.normalised_sentence.values) # only normalise target entities
        elif args.normalise == "all":
            data = list(df.full_normalised_sentence.values) # normalise all entities in sentences
        elif args.normalise == "none":
            data = list(df.sentence.values)
        else:
            raise ValueError("normalised type MUST BE target / all / none.")
        
        if not inference:
            labels = [tuple([int(ll) for ll in l.split()]) for l in df.label.values]
        
        # shuffle the data for training set
        if not inference:
            data = list(zip(data,labels))
            if not eval:
                indices = list(range(len(df)))
                np.random.shuffle(indices)
                data = [data[i] for i in indices]

        data = [data[i:i+args.batch_size] for i in range(0,len(data),args.batch_size)]
        self.data = data
        logger.info(f"{tag}: {len(data)} batches generated.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self,key):
        if not isinstance(key,int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]

        if not self.inference:
            batch_sents, batch_labels = map(list,zip(*batch))
        else:
            batch_sents = batch
        #print(batch_sents) 
        encoding = syntax_process(batch_sents,self.tokenizer,parser,self.device,self.max_len,self.add_const)
        
        if not self.inference:
            batch_labels = mlb.fit_transform(batch_labels).astype(np.float32)
            #print(batch_labels)
            encoding.update({"labels":torch.from_numpy(batch_labels).to(self.device)})
        return encoding

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_wp_to_token(wps):
    word_end_marks = [i for i, wp in enumerate(wps) if not wp.startswith("##")]
    word_end_marks.append(len(wps))
    
    tokens = []
    start = word_end_marks[0]
    for end in word_end_marks[1:]:
        tokens.append(''.join([c.strip("##") for c in wps[start:end]]))
        start = end
    return tokens, word_end_marks

def map_token_to_const(tree,ix):
    if type(tree) == str:
        return [ix,ix+1], False
    
    res = []
    curr_ix = ix
    all_has_p = False
    
    for t in tree:
        if type(t) != str and t.label() in ["NP","VP","ADJP","ADVP","PP"]:
            all_has_p = True
            tmp, has_p = map_token_to_const(t,curr_ix)
            if not has_p:
                res += [tmp[0],tmp[-1]]
            else:
                res += tmp
            #print(tmp,t.label(),has_p)
        else:
            #if type(t) != str:print(t.label())
            tmp, has_p = map_token_to_const(t,curr_ix)
            res += tmp
        #print(res)
        all_has_p = all_has_p or has_p
        curr_ix = tmp[-1]
        
    return sorted(list(set(res))), all_has_p

def merge_maps(wp2token,token2const):
    wp2const = []
    start = token2const[0]
    for end in token2const[1:]:
        wp2const.append(wp2token[start])
        start = end
    wp2const.append(wp2token[-1])
    return wp2const

def syntax_process(sentences,tokenizer,const_parser,device,max_len=256,add_const=False,printout=False):
    encoding = tokenizer(sentences, add_special_tokens = True,    
                         truncation = True, 
                         max_length = max_len,
                         padding = "max_length", 
                         return_attention_mask = True, 
                         return_tensors = "pt")

    encoding =  {k:v.to(device) for k, v in encoding.items()}

    if not add_const:
        return encoding
    
    wps = [[tokenizer.ids_to_tokens[i] for i in ids if i > 0] for ids in (encoding["input_ids"].tolist())]
    tokens, wp2token = zip(*[map_wp_to_token(t[1:-1]) for t in wps])
    
    full_sents = [benepar.InputSentence(words=t) for t in tokens]
    trees = parser.parse_sents(full_sents)
    
    res = []
    for tree, sent, wp, wp2token_tmp in zip(list(trees),tokens,wps,wp2token):
        token2const_tmp = map_token_to_const(tree,0)[0]
        if printout:
            tree.pprint()
            consts = []
            start = token2const_tmp[0]
            for end in token2const_tmp[1:]:
                consts.append(' '.join(sent[start:end]))
                start = end
            print('---'.join(consts))
        wp2const_tmp = merge_maps(wp2token_tmp,token2const_tmp)
        res.append([0]+[ixx + 1 for ixx in wp2const_tmp]+[len(wp)])
    
    encoding["wp2const"] = res
    
    return encoding
