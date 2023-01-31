import torch

mg_path = "../examples/checkpoints/vision_classify/iter_0000001/mp_rank_00/model_optim_rng.pt"
model_mg = torch.load(mg_path, map_location=torch.device('cpu'))
print(type(model_mg))
print(model_mg.keys())
print(model_mg['model'].keys())
# print(model_mg['model']['language_model']['embedding'])
for key in model_mg['model'].keys():
    if key.startswith("backbone.transformer.layers"):
        if key.split(".")[3] == '0':
            print("key", key, model_mg['model'][key].size())
    else:
        print("key", key, model_mg['model'][key].size())

# print(model_mg['model']['binary_head'].keys())
# print(model_mg['iteration'])
# print(model_mg['config'])
print("\033[31m ================================ \033[0m")
hf_path = "../examples/checkpoints/vision_classify/release/mp_rank_00/vit_base_patch16_hf.pt"
model_hf = torch.load(hf_path, map_location=torch.device('cpu'))
print(type(model_hf))
print(model_hf.keys())
for key in model_hf.keys():
    if key.startswith("vit.encoder"):
        if "layer.0" in key or "layer.11" in key:
            print(key, model_hf[key].size())
    else:
        print(key, model_hf[key].size())

state_dict = {}
state_dict['model'] = {}
query_key_value_weight = {}
query_key_value_bias = {}

for key in model_hf.keys():
    if key == "vit.embeddings.cls_token":
        state_dict['model']['backbone.embeddings.cls_token'] = model_hf[key]
    elif key == "vit.embeddings.position_embeddings":
        state_dict['model']['backbone.embeddings.position_embeddings'] = model_hf[key]
    elif key == "vit.embeddings.patch_embeddings.projection.weight":
        state_dict['model']['backbone.embeddings.patch_embeddings.projection.weight'] = model_hf[key]
    elif key == "vit.embeddings.patch_embeddings.projection.bias":
        state_dict['model']['backbone.embeddings.patch_embeddings.projection.bias'] = model_hf[key]
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
        state_dict['model']['backbone.transformer.layers.' + str(key.split('.')[3]) + '.self_attention.query_key_value.weight'] \
            = torch.cat((query_key_value_weight['query'],
                         query_key_value_weight['key'],
                         query_key_value_weight['value']), 0)
        query_key_value_weight.clear()
    elif "value.bias" in key:
        query_key_value_bias['value'] = model_hf[key]
        state_dict['model']['backbone.transformer.layers.' + str(key.split('.')[3]) + '.self_attention.query_key_value.bias'] \
            = torch.cat((query_key_value_bias['query'],
                         query_key_value_bias['key'],
                         query_key_value_bias['value']), 0)
        query_key_value_bias.clear()
    elif "attention.output.dense.weight" in key:
        state_dict['model']['backbone.transformer.layers.' + str(key.split('.')[3]) + '.self_attention.dense.weight'] \
            = model_hf[key]
    elif "attention.output.dense.bias" in key:
        state_dict['model']['backbone.transformer.layers.' + str(key.split('.')[3]) + '.self_attention.dense.bias'] \
            = model_hf[key]
    elif "layernorm_before.weight" in key:
        state_dict['model']['backbone.transformer.layers.' + str(key.split('.')[3]) + '.input_layernorm.weight'] \
            = model_hf[key]
    elif "layernorm_before.bias" in key:
        state_dict['model']['backbone.transformer.layers.' + str(key.split('.')[3]) + '.input_layernorm.bias'] \
            = model_hf[key]
    elif "layernorm_after.weight" in key:
        state_dict['model']['backbone.transformer.layers.' + str(key.split('.')[3]) + '.post_attention_layernorm.weight'] \
            = model_hf[key]
    elif "layernorm_after.bias" in key:
        state_dict['model']['backbone.transformer.layers.' + str(key.split('.')[3]) + '.post_attention_layernorm.bias'] \
            = model_hf[key]
    elif "intermediate.dense.weight" in key:
        state_dict['model']['backbone.transformer.layers.' + str(key.split('.')[3]) + '.mlp.dense_h_to_4h.weight'] \
            = model_hf[key]
    elif "intermediate.dense.bias" in key:
        state_dict['model']['backbone.transformer.layers.' + str(key.split('.')[3]) + '.mlp.dense_h_to_4h.bias'] \
            = model_hf[key]
    elif "output.dense.weight" in key and len(key.split('.')) == 7:
        state_dict['model']['backbone.transformer.layers.' + str(key.split('.')[3]) + '.mlp.dense_4h_to_h.weight'] \
            = model_hf[key]
    elif "output.dense.bias" in key and len(key.split('.')) == 7:
        state_dict['model']['backbone.transformer.layers.' + str(key.split('.')[3]) + '.mlp.dense_4h_to_h.bias'] \
            = model_hf[key]
    elif "vit.layernorm.weight" in key:
        state_dict['model']['backbone.transformer.final_layernorm.weight'] = model_hf[key]
    elif "vit.layernorm.bias" in key:
        state_dict['model']['backbone.transformer.final_layernorm.bias'] = model_hf[key]


torch.save(state_dict, "../examples/checkpoints/vision_classify/release/mp_rank_00/model_optim_rng.pt")
print("\033[31m ================================ \033[0m")
model_check = torch.load("../examples/checkpoints/vision_classify/release/mp_rank_00/model_optim_rng.pt",
                         map_location=torch.device('cpu'))
print(type(model_check))
print(model_check.keys())
print(model_check['model'].keys())
# print(model_check['model']['language_model'].keys())
# print(model_mg['model']['language_model']['embedding'])

for key in model_check['model'].keys():
    if key.startswith("backbone.transformer.layers"):
        if key.split(".")[3] == '0':
            print("key:", key, model_check['model'][key].size())
    else:
        print("key:", key, model_check['model'][key].size())

