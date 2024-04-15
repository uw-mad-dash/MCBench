import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_PATH = "facebook/xlm-roberta-xl"
local_path = "../examples/checkpoints/xlm-roberta-xl/release/mp_rank_00"


model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model_state_dict = model.state_dict()

if not os.path.exists(local_path):
    os.makedirs(local_path)

hf_path = os.path.join(local_path, "xlm_roberta_xl_hf.pt")
torch.save(model_state_dict, hf_path)
print("\033[31m ================================ \033[0m")
model_hf = torch.load(hf_path, map_location=torch.device('cpu'))
print(type(model_hf))
print(model_hf.keys())
for key in model_hf.keys():
    print(key, model_hf[key].size())

state_dict = {}
state_dict['model'] = {}
state_dict['model']['language_model'] = {}
state_dict['model']['language_model']['embedding'] = {}
state_dict['model']['language_model']['transformer'] = {}
state_dict['model']['language_model']['pooler'] = {}
query_key_value_weight = {}
query_key_value_bias = {}

for key in model_hf.keys():
    if key == "roberta.embeddings.word_embeddings.weight":
        state_dict['model']['language_model']['embedding']['word_embeddings'] = {}
        state_dict['model']['language_model']['embedding']['word_embeddings']['weight'] = model_hf[key]
    elif key == "roberta.embeddings.position_embeddings.weight":
        state_dict['model']['language_model']['embedding']['position_embeddings'] = {}
        state_dict['model']['language_model']['embedding']['position_embeddings']['weight'] = model_hf[key]
    elif key == "roberta.embeddings.token_type_embeddings.weight":
        state_dict['model']['language_model']['embedding']['tokentype_embeddings'] = {}
        state_dict['model']['language_model']['embedding']['tokentype_embeddings']['weight'] \
            = torch.cat((model_hf[key], model_hf[key]), dim=0)
    elif "self_attn_layer_norm.weight" in key:
        layer_num = key.split('.')[3]
        state_dict['model']['language_model']['transformer'][f'layers.{layer_num}.input_layernorm.weight'] \
            = model_hf[key]
    elif "self_attn_layer_norm.bias" in key:
        layer_num = key.split('.')[3]
        state_dict['model']['language_model']['transformer'][f'layers.{layer_num}.input_layernorm.bias'] \
            = model_hf[key]
    elif "query.weight" in key:
        query_key_value_weight['query'] = model_hf[key]
    elif "query.bias" in key:
        query_key_value_bias['query'] = model_hf[key]
    elif "key.weight" in key:
        query_key_value_weight['key'] = model_hf[key]
    elif "key.bias" in key:
        query_key_value_bias['key'] = model_hf[key]
    elif "value.weight" in key:
        layer_num = key.split('.')[3]
        query_key_value_weight['value'] = model_hf[key]
        state_dict['model']['language_model']['transformer'][f'layers.{layer_num}.attention.query_key_value.weight'] \
            = torch.cat((query_key_value_weight['query'],
                         query_key_value_weight['key'],
                         query_key_value_weight['value']), 0)
        query_key_value_weight.clear()
    elif "value.bias" in key:
        layer_num = key.split('.')[3]
        query_key_value_bias['value'] = model_hf[key]
        state_dict['model']['language_model']['transformer'][f'layers.{layer_num}.attention.query_key_value.bias'] \
            = torch.cat((query_key_value_bias['query'],
                         query_key_value_bias['key'],
                         query_key_value_bias['value']), 0)
        query_key_value_bias.clear()
    elif "attention.output.dense.weight" in key:
        layer_num = key.split('.')[3]
        state_dict['model']['language_model']['transformer'][f'layers.{layer_num}.attention.dense.weight'] \
            = model_hf[key]
    elif "attention.output.dense.bias" in key:
        layer_num = key.split('.')[3]
        state_dict['model']['language_model']['transformer'][f'layers.{layer_num}.attention.dense.bias'] \
            = model_hf[key]
    elif "LayerNorm.weight" in key and len(key.split('.')) == 6:
        layer_num = key.split('.')[3]
        state_dict['model']['language_model']['transformer'][f'layers.{layer_num}.post_attention_layernorm.weight'] \
            = model_hf[key]
    elif "LayerNorm.bias" in key and len(key.split('.')) == 6:
        layer_num = key.split('.')[3]
        state_dict['model']['language_model']['transformer'][f'layers.{layer_num}.post_attention_layernorm.bias'] \
            = model_hf[key]
    elif "intermediate.dense.weight" in key:
        layer_num = key.split('.')[3]
        state_dict['model']['language_model']['transformer'][f'layers.{layer_num}.mlp.dense_h_to_4h.weight'] \
            = model_hf[key]
    elif "intermediate.dense.bias" in key:
        layer_num = key.split('.')[3]
        state_dict['model']['language_model']['transformer'][f'layers.{layer_num}.mlp.dense_h_to_4h.bias'] \
            = model_hf[key]
    elif "output.dense.weight" in key:
        layer_num = key.split('.')[3]
        state_dict['model']['language_model']['transformer'][f'layers.{layer_num}.mlp.dense_4h_to_h.weight'] \
            = model_hf[key]
    elif "output.dense.bias" in key:
        layer_num = key.split('.')[3]
        state_dict['model']['language_model']['transformer'][f'layers.{layer_num}.mlp.dense_4h_to_h.bias'] \
            = model_hf[key]
    elif key == "roberta.encoder.LayerNorm.weight":
        state_dict['model']['language_model']['transformer']['final_layernorm.weight'] = model_hf[key]
    elif key == "roberta.encoder.LayerNorm.bias":
        state_dict['model']['language_model']['transformer']['final_layernorm.bias'] = model_hf[key]
    elif key == "classifier.dense.weight":
        state_dict['model']['language_model']['pooler']['dense.weight'] = model_hf[key]
    elif key == "classifier.dense.bias":
        state_dict['model']['language_model']['pooler']['dense.bias'] = model_hf[key]


save_path = os.path.join(local_path, "model_optim_rng.pt")
torch.save(state_dict, save_path)
print("\033[31m ================================ \033[0m")
model_check = torch.load(save_path, map_location=torch.device('cpu'))
print(type(model_check))
print(model_check.keys())
print(model_check['model'].keys())
print(model_check['model']['language_model'].keys())
for key_first in model_check['model']['language_model']['embedding'].keys():
    for key_second in model_check['model']['language_model']['embedding'][key_first].keys():
        print("embedding", key_first, key_second,
              model_check['model']['language_model']['embedding'][key_first][key_second].size())

for key_first in model_check['model']['language_model']['transformer'].keys():
    # if "layers.0" in key_first:
    print("transformer", key_first,
          model_check['model']['language_model']['transformer'][key_first].size())

for key_first in model_check['model']['language_model']['pooler'].keys():
    print("pooler", key_first,
          model_check['model']['language_model']['pooler'][key_first].size())




