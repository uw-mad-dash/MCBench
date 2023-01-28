import torch
import re

model_mg = torch.load("examples/checkpoints/bert_345m/release/mp_rank_00/model_optim_rng.pt",
                      map_location=torch.device('cpu'))
print(type(model_mg))
print(model_mg.keys())
print(model_mg['model'].keys())
print(model_mg['model']['language_model'].keys())
print(model_mg['model']['lm_head'].keys())
print(model_mg['model']['binary_head'].keys())
print(model_mg['iteration'])
print(model_mg['config'])
# print(model_mg['model']['language_model']['embedding'])
for key_first in model_mg['model']['language_model']['embedding'].keys():
    for key_second in model_mg['model']['language_model']['embedding'][key_first].keys():
        print("embedding", key_first, key_second,
              model_mg['model']['language_model']['embedding'][key_first][key_second].size())

for key_first in model_mg['model']['language_model']['transformer'].keys():
    # if "layers.0" in key_first:
    print("transformer", key_first,
          model_mg['model']['language_model']['transformer'][key_first].size())

for key_first in model_mg['model']['language_model']['pooler'].keys():
    print("pooler", key_first,
          model_mg['model']['language_model']['pooler'][key_first].size())

print(model_mg['model']['lm_head'].keys())
for key_first in model_mg['model']['lm_head'].keys():
    print("lm_head", key_first, model_mg['model']['lm_head'][key_first].size())

print(model_mg['model']['binary_head'].keys())
for key_first in model_mg['model']['binary_head'].keys():
    print("binary_head", key_first, model_mg['model']['binary_head'][key_first].size())

# print(model_mg['model']['binary_head'].keys())
# print(model_mg['iteration'])
# print(model_mg['config'])
print("\033[31m ================================ \033[0m")
model_nv = torch.load("examples/checkpoints/bert_base_nv/release/mp_rank_00/bert_base.pt",
                      map_location=torch.device('cpu'))
print(type(model_nv))
print(model_nv.keys())
print(model_nv['model'].keys())
for key in model_nv['model'].keys():
    if key.startswith("encoder"):
        # if "layer.0" in key or "layer.11" in key:
        print(key, model_nv['model'][key].size(), torch.flatten(model_nv['model'][key])[0])
    else:
        print(key, model_nv['model'][key].size(), torch.flatten(model_nv['model'][key])[0])

state_dict = {}
state_dict['model'] = {}
state_dict['model']['language_model'] = {}
state_dict['model']['language_model']['embedding'] = {}
state_dict['model']['language_model']['transformer'] = {}
state_dict['model']['language_model']['pooler'] = {}
state_dict['model']['lm_head'] = {}
state_dict['model']['binary_head'] = {}
state_dict['iteration'] = 2000000
state_dict['config'] = {'hidden-size': 768, 'num-attention-heads': 12, 'num-layers': 12, 'max-seq-length': 128}
query_key_value_weight = {}
query_key_value_bias = {}

for key in model_nv['model'].keys():
    if key == "bert.embeddings.word_embeddings.weight":
        state_dict['model']['language_model']['embedding']['word_embeddings'] = {}
        state_dict['model']['language_model']['embedding']['word_embeddings']['weight'] = model_nv['model'][key]
    elif key == "bert.embeddings.position_embeddings.weight":
        state_dict['model']['language_model']['embedding']['position_embeddings'] = {}
        state_dict['model']['language_model']['embedding']['position_embeddings']['weight'] = model_nv['model'][key]
    elif key == "bert.embeddings.token_type_embeddings.weight":
        state_dict['model']['language_model']['embedding']['tokentype_embeddings'] = {}
        state_dict['model']['language_model']['embedding']['tokentype_embeddings']['weight'] = model_nv['model'][key]
    elif key == "bert.embeddings.LayerNorm.weight":
        state_dict['model']['language_model']['transformer']['layers.0.input_layernorm.weight'] = model_nv['model'][key]
    elif key == "bert.embeddings.LayerNorm.bias":
        state_dict['model']['language_model']['transformer']['layers.0.input_layernorm.bias'] = model_nv['model'][key]
    elif "query.weight" in key:
        # print("\033[31m query \033[0m", key.split('.')[3])
        query_key_value_weight['query'] = model_nv['model'][key]
    elif "query.bias" in key:
        query_key_value_bias['query'] = model_nv['model'][key]
    elif "key.weight" in key:
        # print("\033[31m key \033[0m", key.split('.')[3])
        query_key_value_weight['key'] = model_nv['model'][key]
    elif "key.bias" in key:
        query_key_value_bias['key'] = model_nv['model'][key]
    elif "value.weight" in key:
        # print("\033[31m value \033[0m", key.split('.')[3])
        query_key_value_weight['value'] = model_nv['model'][key]
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.attention.query_key_value.weight'] \
            = torch.transpose(torch.cat((query_key_value_weight['query'],
                                         query_key_value_weight['key'],
                                         query_key_value_weight['value']), 1), 0, 1)
        # state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.attention.query_key_value.weight'] \
        #     = torch.cat((torch.transpose(query_key_value_weight['query'], 0, 1),
        #                  torch.transpose(query_key_value_weight['key'], 0, 1),
        #                  torch.transpose(query_key_value_weight['value'], 0, 1)), 0)
        query_key_value_weight.clear()
    elif "value.bias" in key:
        query_key_value_bias['value'] = model_nv['model'][key]
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.attention.query_key_value.bias'] \
            = torch.cat((query_key_value_bias['query'],
                         query_key_value_bias['key'],
                         query_key_value_bias['value']), 0)
        query_key_value_bias.clear()
    elif "attention.output.dense.weight" in key:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.attention.dense.weight'] \
            = model_nv['model'][key]
    elif "attention.output.dense.bias" in key:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.attention.dense.bias'] \
            = model_nv['model'][key]
    elif "attention.output.LayerNorm.weight" in key:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.post_attention_layernorm.weight'] \
            = model_nv['model'][key]
    elif "attention.output.LayerNorm.bias" in key:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.post_attention_layernorm.bias'] \
            = model_nv['model'][key]
    elif "intermediate.dense_act.weight" in key:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.mlp.dense_h_to_4h.weight'] \
            = model_nv['model'][key]
    elif "intermediate.dense_act.bias" in key:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.mlp.dense_h_to_4h.bias'] \
            = model_nv['model'][key]
    elif "output.dense.weight" in key and len(key.split('.')) == 7:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.mlp.dense_4h_to_h.weight'] \
            = model_nv['model'][key]
    elif "output.dense.bias" in key and len(key.split('.')) == 7:
        state_dict['model']['language_model']['transformer']['layers.' + str(key.split('.')[3]) + '.mlp.dense_4h_to_h.bias'] \
            = model_nv['model'][key]
    elif "output.LayerNorm.weight" in key:
        layer_num = int(key.split('.')[3]) + 1
        if layer_num == 12:
            state_dict['model']['language_model']['transformer']['final_layernorm.weight'] = model_nv['model'][key]
        else:
            state_dict['model']['language_model']['transformer']['layers.' + str(layer_num) + '.input_layernorm.weight'] \
                = model_nv['model'][key]
    elif "output.LayerNorm.bias" in key:
        layer_num = int(key.split('.')[3]) + 1
        if layer_num == 12:
            state_dict['model']['language_model']['transformer']['final_layernorm.bias'] = model_nv['model'][key]
        else:
            state_dict['model']['language_model']['transformer']['layers.' + str(layer_num) + '.input_layernorm.bias'] \
                = model_nv['model'][key]
    elif key == "bert.pooler.dense_act.weight":
        state_dict['model']['language_model']['pooler']['dense.weight'] = model_nv['model'][key]
    elif key == "bert.pooler.dense_act.bias":
        state_dict['model']['language_model']['pooler']['dense.bias'] = model_nv['model'][key]
    elif key == "cls.predictions.bias":
        state_dict['model']['lm_head']['bias'] = model_nv['model'][key]
    elif key == "cls.predictions.transform.dense_act.weight":
        state_dict['model']['lm_head']['dense.weight'] = model_nv['model'][key]
    elif key == "cls.predictions.transform.dense_act.bias":
        state_dict['model']['lm_head']['dense.bias'] = model_nv['model'][key]
    elif key == "cls.predictions.transform.LayerNorm.weight":
        state_dict['model']['lm_head']['layernorm.weight'] = model_nv['model'][key]
    elif key == "cls.predictions.transform.LayerNorm.bias":
        state_dict['model']['lm_head']['layernorm.bias'] = model_nv['model'][key]
    elif key == "cls.seq_relationship.weight":
        state_dict['model']['binary_head']['weight'] = model_nv['model'][key]
    elif key == "cls.seq_relationship.bias":
        state_dict['model']['binary_head']['bias'] = model_nv['model'][key]

# state_dict['checkpoint_version'] = 3.0

torch.save(state_dict, "examples/checkpoints/bert_base_nv/release/mp_rank_00/model_optim_rng.pt")
print("\033[31m ================================ \033[0m")
model_check = torch.load("examples/checkpoints/bert_base_nv/release/mp_rank_00/model_optim_rng.pt",
                         map_location=torch.device('cpu'))
print(type(model_check))
print(model_check.keys())
print(model_check['model'].keys())
print(model_check['model']['language_model'].keys())
print(model_check['model']['language_model']['embedding'].keys())

for key_first in model_check['model']['language_model']['embedding'].keys():
    for key_second in model_check['model']['language_model']['embedding'][key_first].keys():
        print("embedding", key_first, key_second,
              model_check['model']['language_model']['embedding'][key_first][key_second].size(),
              torch.flatten(model_check['model']['language_model']['embedding'][key_first][key_second])[0])

for key_first in model_check['model']['language_model']['transformer'].keys():
    # if "layers.0" in key_first:
    print("transformer", key_first,
          model_check['model']['language_model']['transformer'][key_first].size(),
          torch.flatten(model_check['model']['language_model']['transformer'][key_first])[0])

for key_first in model_check['model']['language_model']['pooler'].keys():
    print("pooler", key_first,
          model_check['model']['language_model']['pooler'][key_first].size(),
          torch.flatten(model_check['model']['language_model']['pooler'][key_first])[0])

# print(model_mg['model']['lm_head'].keys())
for key_first in model_check['model']['lm_head'].keys():
    print("lm_head", key_first, model_check['model']['lm_head'][key_first].size(),
          torch.flatten(model_check['model']['lm_head'][key_first])[0])

# print(model_mg['model']['binary_head'].keys())
for key_first in model_check['model']['binary_head'].keys():
    print("binary_head", key_first, model_check['model']['binary_head'][key_first].size(),
          torch.flatten(model_check['model']['binary_head'][key_first])[0])
