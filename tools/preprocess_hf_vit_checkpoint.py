import os
import torch
from transformers import ViTForImageClassification

model_name = "vit_base_patch32"
if model_name == "vit_base_patch16":
    pretrain_name = "google/vit-base-patch16-224-in21k"
    local_path = "../examples/checkpoints/vision_classify_base_patch16/release/mp_rank_00"
elif model_name == "vit_base_patch32":
    pretrain_name = "google/vit-base-patch32-224-in21k"
    local_path = "../examples/checkpoints/vision_classify_base_patch32/release/mp_rank_00"
elif model_name == "vit_large_patch16":
    pretrain_name = "google/vit-large-patch16-224-in21k"
    local_path = "../examples/checkpoints/vision_classify_large_patch16/release/mp_rank_00"
elif model_name == "vit_large_patch32":
    pretrain_name = "google/vit-large-patch32-224-in21k"
    local_path = "../examples/checkpoints/vision_classify_large_patch32/release/mp_rank_00"
elif model_name == "vit_huge_patch14":
    pretrain_name = "google/vit-huge-patch14-224-in21k"
    local_path = "../examples/checkpoints/vision_classify_huge_patch14/release/mp_rank_00"
else:
    raise ValueError("model name is wrong")

vit_model = ViTForImageClassification.from_pretrained(pretrain_name)

# print(vit_base_16.state_dict().keys())
vit_model_state_dict = vit_model.state_dict()

# for key in vit_model_state_dict.keys():
#     print("key: ", key, " shape: ", vit_model_state_dict[key].size())

if not os.path.exists(local_path):
    os.makedirs(local_path)
print("\033[31m ================================ \033[0m")
hf_path = os.path.join(local_path, model_name + "_hf.pt")
torch.save(vit_model_state_dict, hf_path)
# hf_path = "../examples/checkpoints/vision_classify_large_patch32/release/mp_rank_00/vit_large_patch32_hf.pt"
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

save_path = os.path.join(local_path, "model_optim_rng.pt")
torch.save(state_dict, save_path)
# torch.save(state_dict, "../examples/checkpoints/vision_classify_large_patch32/release/mp_rank_00/model_optim_rng.pt")
print("\033[31m ================================ \033[0m")

model_check = torch.load(save_path, map_location=torch.device('cpu'))
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

