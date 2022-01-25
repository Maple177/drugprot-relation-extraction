from argparse import ArgumentParser


def get_args():
     parser = ArgumentParser(description='Finetuning BERT models')
     
     parser.add_argument("--data_dir", default=None, type=str, required=True,
                         help="The training data path. Should contain the .csv files")
     parser.add_argument("--dev_data_dir", default=None, type=str, 
                         help="The validation data path. Should contain the .csv files")
     parser.add_argument("--train_dev_split",default=0.2,type=float,
                         help="Ratio of validation set for the train-validation-split")
     parser.add_argument("--force_cpu",action="store_true",
                         help="if set, the script will be run WITHOUT GPU.")
     parser.add_argument("--bert_variant",default="biobert",type=str,
                         help="abbreviation of model to use")
     parser.add_argument("--pretrained_model_path", default=None, type=str,
                         help="Path to pre-trained model, if no pre-trained model the model that corresponds to the model type "
                              "will be dowloaded to the path.")
     parser.add_argument("--config_name_or_path", default="", type=str, 
                         help="Path to pre-trained config or shortcut name selected in the list")
     parser.add_argument("--finetuned_model_path", default=None, type=str, 
                         help="The output directory where the model predictions and checkpoints will be written.")
     parser.add_argument("--output_dir",default=None,type=str,
                         help="Path to predictions (set it only for inference)")
     parser.add_argument("--binary",action="store_true",
                         help="if set, relation type will not be considered.")
     parser.add_argument("--num_labels",type=int,default=2,
                         help="number of relations (no_relation COUNTED)")
     parser.add_argument("--debug",action="store_true",
                         help="use only first 100 examples to test the whole script.")
     parser.add_argument("--num_debug",type=int,default=100,
                         help="first num_debug examples will be used for a fast test.")
     parser.add_argument("--seed",type=int,default=42,
                         help="random seed for ensure reproducibility")
     parser.add_argument("--batch_size", default=32, type=int,
                         help="Batch size per GPU/CPU for training.")

     group = parser.add_argument_group('--syntax_options')
     parser.add_argument("--model_type",type=str,required=True,help="type of model; MUST BE in {no_syntax, chunking, const_tree}.")
     group.add_argument("--num_syntax_layers",type=int,default=2,help="number of BERT layers to add to "
                                                                      "integrate the syntactic information.")     

     group = parser.add_argument_group('--training_options')
     group.add_argument('--logging_steps', type=int, default=50,
                         help="Log every X updates steps.")
     group.add_argument("--monitor",type=str,default="score",
                         help="criteria to use for early stopping")
     group.add_argument("--early_stopping",action="store_true",
                         help="if use early stopping during training")
     group.add_argument("--max_num_epochs",default=10,type=int,
                         help="maximum number of epochs")
     group.add_argument("--num_train_epochs", default=3, type=int,
                         help="Total number of training epochs to perform.")
     group.add_argument("--patience",type=int,default=3,
                         help="patience of early stopping")
     group.add_argument("--max_seq_length", default=128, type=int,
                         help="The maximum input sequence length after tokenization. Sequences longer "
                              "than this will be truncated, sequences shorter will be padded.")
     group.add_argument("--num_ensemble",type=int,
                         help="number of repetitive experiments to get an ensemble result")
     group.add_argument("--learning_rate",type=float,default=2e-5)
     group.add_argument("--max_grad_norm", default=1.0, type=float,
                         help="Max gradient norm for gradient clipping.")
     
     args = parser.parse_args()
     return args
