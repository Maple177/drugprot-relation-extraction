# DrugProt-drug-chemical-protein-relation-extraction
This repository includes pytorch codes for using domain-specific BERT models, along with proposed const-BERT variants on the Drugprot track of BioCreative VII Challenge.

## Data
We extracted and pre-processed the original dataset from the official DrugProt task website:
https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-1/

The processed data is converted to several .csv files, for convenience, we make these processed files public and downloadable from:
https://drive.google.com/file/d/1piprXA0QbsKvcloxLUQFkg0rpFXPGr74/view?usp=sharing

## Train & Make Predictions
Download codes and data:
```
git clone https://github.com/Maple177/drugprot-relation-extraction.git
cd ./drugprot-relation-extraction/
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1piprXA0QbsKvcloxLUQFkg0rpFXPGr74' -O data.tgz
tar xzvf data.tgz
rm data.tgz
```
Make sure you have available GPU before training;
below we show example commands for training an ensemble containing 8 bioBERT model using batch size=16, maximum sequence length=256, and early stopping enabled:

train (e.g. original bioBERT) 
```
python3 train.py --data_dir ./data/train.csv --dev_data_dir ./data/dev.csv --model_type biobert --pretrained_model_path ./pretrained_model_biobert/ --finetuned_model_path ./finetuned_model_biobert/  --num_labels 14 --num_ensemble 8 --batch_size 16 --max_seq_length 256 --max_num_epochs 10 --no_randomness --early_stopping --normalise none
```
inference on the test set
```
python3 eval.py --data_dir ./data/test.csv --output_dir ./prediction_biobert/testing/ --model_type biobert --pretrained_model_path ./pretrained_model_biobert/ --finetuned_model_path ./finetuned_model_biobert/  --num_labels 14 --num_ensemble 8 --batch_size 16 --max_seq_length 256 --normalise none
```

make .tsv submission files (in the format required by organisers)
```
python3 make_submission.py --data_dir ./dev.csv --pred_dir ./prediction_biobert/testing/ --id2label_dir ./data/ --output_dir ./submission_biobert.tsv --num_ensemble 8
```
