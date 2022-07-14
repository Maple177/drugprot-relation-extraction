#!bin/sh

#BioBERT
python3 train.py --data_dir ./data/ --bert_variant biobert --model_type no_syntax_extra --pretrained_model_path ./pretrained_models/ --finetuned_model_path ./models/ --num_labels 14 --max_num_epochs 20 --num_ensemble 5 --seed 42 --early_stopping --monitor score --learning_rate 2e-05 --batch_size 16 --grid_search --run_id 8
python3 eval.py --data_dir ./data/ --bert_variant biobert --model_type no_syntax_extra --pretrained_model_path ./pretrained_models/ --finetuned_model_path ./models/ --num_labels 14 --num_ensemble 5 --seed 42 --batch_size 16 --learning_rate 2e-05 --run_id 80


#CE-BioBERT
python3 drugprot-relation-extraction/train.py --data_dir ./data/ --bert_variant biobert --model_type with_chunking --pretrained_model_path ./pretrained_models/ --finetuned_model_path ./models/ --num_labels 14 --max_num_epochs 20 --num_ensemble 5 --seed 42 --early_stopping --monitor score --learning_rate 2e-05 --num_syntax_layers 2 --batch_size 16 --grid_search --run_id 3
python3 eval.py --data_dir ./data/ --bert_variant biobert --model_type with_chunking --pretrained_model_path ./pretrained_models/ --finetuned_model_path ./models/ --num_labels 14 --num_ensemble 5 --seed 42  --batch_size 16 --learning_rate 2e-05 --num_syntax_layers 2 --run_id 4

#CT-BioBERT
python3 train.py --data_dir ./data/ --bert_variant biobert --model_type with_const_tree --pretrained_model_path ./pretrained_models/ --finetuned_model_path ./models/ --num_labels 14 --max_num_epochs 20 --num_ensemble 5 --seed 42 --early_stopping --monitor score --learning_rate 2e-05 --batch_size 16 --grid_search --run_id 5
python3 eval.py --data_dir ./data --bert_variant biobert --model_type with_const_tree --pretrained_model_path ./pretrained_models/ --finetuned_model_path ./models/ --num_labels 14 --num_ensemble 5 --seed 42  --batch_size 16 --learning_rate 2e-05 --run_id 6

#Late-Fusion
python3 train.py --data_dir ./data/ --bert_variant biobert --model_type late_fusion --pretrained_model_path ./pretrained_models/ --finetuned_model_path ./models/ --num_labels 14 --max_num_epochs 20 --num_ensemble 5 --seed 42 --early_stopping --monitor score --learning_rate 1e-05 --num_syntax_layers 4 --batch_size 16 --grid_search --run_id 7
python3 eval.py --data_dir ./data/ --bert_variant biobert --model_type late_fusion --pretrained_model_path ./pretrained_models/ --finetuned_model_path ./models/ --num_labels 14 --num_ensemble 5 --seed 42  --batch_size 16 --learning_rate 1e-05 --num_syntax_layers 4 --run_id 8
