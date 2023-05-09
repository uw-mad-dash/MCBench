import os
import torch
from transformers import BertForSequenceClassification

pretrain_name = "bert-base-cased"
local_path = "../examples/checkpoints/bert_base_hf/release/mp_rank_00"

bert_model = BertForSequenceClassification.from_pretrained(pretrain_name)

bert_model_state_dict = bert_model.state_dict()

if not os.path.exists(local_path):
    os.makedirs(local_path)

hf_path = os.path.join(local_path, "bert_base_hf.pt")
torch.save(bert_model_state_dict, hf_path)
print("\033[31m ================================ \033[0m")
model_hf = torch.load(hf_path, map_location=torch.device('cpu'))
print(type(model_hf))
print(model_hf.keys())
for key in model_hf.keys():
    if key.startswith("encoder"):
        # if "layer.0" in key or "layer.11" in key:
        print(key, model_hf[key].size())
    else:
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
    if key == "bert.embeddings.word_embeddings.weight":
        state_dict['model']['language_model']['embedding']['word_embeddings'] = {}
        state_dict['model']['language_model']['embedding']['word_embeddings']['weight'] = model_hf[key]
    elif key == "bert.embeddings.position_embeddings.weight":
        state_dict['model']['language_model']['embedding']['position_embeddings'] = {}
        state_dict['model']['language_model']['embedding']['position_embeddings']['weight'] = model_hf[key]
    elif key == "bert.embeddings.token_type_embeddings.weight":
        state_dict['model']['language_model']['embedding']['tokentype_embeddings'] = {}
        state_dict['model']['language_model']['embedding']['tokentype_embeddings']['weight'] = model_hf[key]
    elif key == "bert.embeddings.LayerNorm.weight":
        state_dict['model']['language_model']['transformer']['layers.0.input_layernorm.weight'] = model_hf[key]
    elif key == "bert.embeddings.LayerNorm.bias":
        state_dict['model']['language_model']['transformer']['layers.0.input_layernorm.bias'] = model_hf[key]
    elif "query.weight" in key:
        query_key_value_weight['query'] = model_hf[key]
    elif "query.bias" in key:
        query_key_value_bias['query'] = model_hf[key]
    elif "key.weight" in key:
        query_key_value_weight['key'] = model_hf[key]
    elif "key.bias" in key:
        query_key_value_bias['key'] = model_hf[key]
    elif "value.weight" in key:
        query_key_value_weight['value'] = model_hf[key]
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.attention.query_key_value.weight'] \
            = torch.cat((query_key_value_weight['query'],
                         query_key_value_weight['key'],
                         query_key_value_weight['value']), 0)
        query_key_value_weight.clear()
    elif "value.bias" in key:
        query_key_value_bias['value'] = model_hf[key]
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.attention.query_key_value.bias'] \
            = torch.cat((query_key_value_bias['query'],
                         query_key_value_bias['key'],
                         query_key_value_bias['value']), 0)
        query_key_value_bias.clear()
    elif "attention.output.dense.weight" in key:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.attention.dense.weight'] \
            = model_hf[key]
    elif "attention.output.dense.bias" in key:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.attention.dense.bias'] \
            = model_hf[key]
    elif "attention.output.LayerNorm.weight" in key:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.post_attention_layernorm.weight'] \
            = model_hf[key]
    elif "attention.output.LayerNorm.bias" in key:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.post_attention_layernorm.bias'] \
            = model_hf[key]
    elif "intermediate.dense.weight" in key:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.mlp.dense_h_to_4h.weight'] \
            = model_hf[key]
    elif "intermediate.dense.bias" in key:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.mlp.dense_h_to_4h.bias'] \
            = model_hf[key]
    elif "output.dense.weight" in key and len(key.split('.')) == 7:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.mlp.dense_4h_to_h.weight'] \
            = model_hf[key]
    elif "output.dense.bias" in key and len(key.split('.')) == 7:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.mlp.dense_4h_to_h.bias'] \
            = model_hf[key]
    elif "output.LayerNorm.weight" in key:
        print(key)
        layer_num = int(key.split('.')[3]) + 1
        if layer_num == 12:
            state_dict['model']['language_model']['transformer']['final_layernorm.weight'] = model_hf[key]
        else:
            state_dict['model']['language_model']['transformer']['layers.' + str(layer_num) + '.input_layernorm.weight'] \
                = model_hf[key]
    elif "output.LayerNorm.bias" in key:
        layer_num = int(key.split('.')[3]) + 1
        if layer_num == 12:
            state_dict['model']['language_model']['transformer']['final_layernorm.bias'] = model_hf[key]
        else:
            state_dict['model']['language_model']['transformer']['layers.' + str(layer_num) + '.input_layernorm.bias'] \
                = model_hf[key]
    elif key == "bert.pooler.dense.weight":
        state_dict['model']['language_model']['pooler']['dense.weight'] = model_hf[key]
    elif key == "bert.pooler.dense.bias":
        state_dict['model']['language_model']['pooler']['dense.bias'] = model_hf[key]

save_path = os.path.join(local_path, "model_optim_rng.pt")
torch.save(state_dict, save_path)
print("\033[31m ================================ \033[0m")
model_check = torch.load(save_path, map_location=torch.device('cpu'))
print(type(model_check))
print(model_check.keys())
print(model_check['model'].keys())
print(model_check['model']['language_model'].keys())
# print(model_mg['model']['language_model']['embedding'])
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


# print("\033[31m ================================ \033[0m")
# origin_local_path = "../examples/checkpoints/bert_base_hf/release/mp_rank_00"
# origin_save_path = os.path.join(origin_local_path, "model_optim_rng.pt")
# model_check = torch.load(origin_save_path, map_location=torch.device('cpu'))
# print(type(model_check))
# print(model_check.keys())
# print(model_check['model'].keys())
# print(model_check['model']['language_model'].keys())
# # print(model_mg['model']['language_model']['embedding'])
# for key_first in model_check['model']['language_model']['embedding'].keys():
#     for key_second in model_check['model']['language_model']['embedding'][key_first].keys():
#         print("embedding", key_first, key_second,
#               model_check['model']['language_model']['embedding'][key_first][key_second].size())
#
# for key_first in model_check['model']['language_model']['transformer'].keys():
#     # if "layers.0" in key_first:
#     print("transformer", key_first,
#           model_check['model']['language_model']['transformer'][key_first].size())
#
# for key_first in model_check['model']['language_model']['pooler'].keys():
#     print("pooler", key_first,
#           model_check['model']['language_model']['pooler'][key_first].size())


