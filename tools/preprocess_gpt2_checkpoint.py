import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



# MODEL_PATH = "xlnet/xlnet-large-cased"
# local_path = "../examples/checkpoints/xlnet-large-cased/release/mp_rank_00"
#
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
# tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-large-cased")
# model_state_dict = model.state_dict()
#
# if not os.path.exists(local_path):
#     os.makedirs(local_path)
#
# hf_path = os.path.join(local_path, "xlnet-large-cased.pt")
# torch.save(model_state_dict, hf_path)
# print("\033[31m ================================ \033[0m")
# model_hf = torch.load(hf_path, map_location=torch.device('cpu'))
# print(type(model_hf))
# print(model_hf.keys())
# for key in model_hf.keys():
#     print(key, model_hf[key].size())

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
# tokenizer.save_pretrained("llama-7b-tokenizer")


save_path = "/users/Master/Megatron-LM/examples/checkpoints/gpt_345m/release/mp_rank_00/model_optim_rng.pt"

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




