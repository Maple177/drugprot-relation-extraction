import logging
import torch
from torch import nn
from torch.nn import (BCELoss, BCEWithLogitsLoss)
from transformers import (BertPreTrainedModel, BertModel, BertLayer)
from transformers.models.roberta.modeling_roberta import (RobertaPreTrainedModel, RobertaModel, RobertaClassificationHead)

logger = logging.getLogger(__name__)
class_weights = [1.358,45.340,98.397,2232.586,4980.385,66.610,28.814,48.717,46.985,12.023,73.158,70.375,32.324,2697.708] 

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertEncoder(nn.Module):
    def __init__(self, config, num_hidden_layers):
        super().__init__()
        self.config = config
        #print(f"NUM_SYNTAX_LAYERS={num_hidden_layers}")
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        for i, layer_module in enumerate(self.layers):
            #if output_hidden_states:
            #    all_hidden_states = all_hidden_states + (hidden_states,)

            #layer_head_mask = head_mask[i] if head_mask is not None else None
            #print(hidden_states.shape,attention_mask.shape)
            layer_outputs = layer_module(hidden_states,attention_mask=attention_mask)
            hidden_states = layer_outputs[0]

        return (hidden_states,)

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self,config,model_type,num_syntax_layers=2):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model_type = model_type

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if model_type == "with_chunking":
            self.extra_bert = BertEncoder(config,num_syntax_layers)
        self.pooler = BertPooler(config) 
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = BCEWithLogitsLoss(pos_weight=torch.Tensor(class_weights))       
        logger.info(f"LOADED: model type-{model_type}; number of extra layers-{num_syntax_layers}")

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        wp2const=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        #print(input_ids.shape,attention_mask.shape)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if self.model_type == "no_syntax":
            pooled_output = outputs[1]
        elif self.model_type == "with_chunking":
            device = next(self.parameters()).device
            hidden_states = outputs[0]
            bs, L, _ = hidden_states.shape
            #regroup wordpiece embeddings to obtain constituent embeddings
            const_masks = torch.zeros(bs,L,device=device)
            const_states = torch.zeros_like(hidden_states)
            for ix, hs, ma in zip(range(bs),hidden_states,wp2const):
                start = 0
                index = 0
                for end in ma:
                    const_states[ix][index] = hs[start:end].mean(axis=0)
                    index += 1
                    start = end
                const_masks[ix][:index] = 1
            const_states.to(device)
            #const_masks = self.get_extended_attention_mask(const_masks,(bs,L),device)
            const_masks = const_masks[:,None,None,:]
            #print(const_states.shape,const_masks.shape)
            #const_masks.to(device)
            #print(device,const_states.device,const_masks.device)
            const_hidden_states = self.extra_bert(const_states,attention_mask=const_masks)[0]
            pooled_output = self.pooler(const_hidden_states)
          
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1,self.num_labels))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

