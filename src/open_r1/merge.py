import os

import torch
import argparse


import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# def compare_model_weights(model1, model2):
#     for name1, param1 in model1.named_parameters():
#         if name1 in model2.state_dict():
#             param2 = model2.state_dict()[name1]
#             if not torch.allclose(param1, param2, rtol=1e-03, atol=1e-03):
#                 print(f"Layer '{name1}': Weights are DIFFERENT.")
#                 return True
#         else:
#             print(f"Layer '{name1}' not found in the second model.")
#             return True
#     return False

def compare_model_weights(model1, model2):
    params1 = {name: param.clone().detach() for name, param in model1.named_parameters()}
    params2 = {name: param.clone().detach() for name, param in model2.named_parameters()}
    are_same = True
    for name in params1.keys():
        if name not in params2:
            print(f"Parameter {name} not found in model2")
            are_same = False
            break
        if not torch.allclose(params1[name], params2[name], atol=1e-6):
            print(f"Parameter {name} differs")
            are_same = False
            break

    return are_same
    # state_dict1 = model1.state_dict()
    # state_dict2 = model2.state_dict()

    # all_keys = set(state_dict1.keys()) & set(state_dict2.keys())
    # different = False

    # for key in sorted(all_keys):
    #     param1 = state_dict1[key]
    #     param2 = state_dict2[key]

    #     if not torch.allclose(param1, param2, rtol=1e-03, atol=1e-03):
    #         print(f"❗ Layer '{key}': Weights are DIFFERENT.")
    #         different = True

    # if not different:
    #     print("✅ All parameters are identical.")
    # return different


def merge_lora_into_base(
    base_model_dir: str,
    lora_model_dir: str,
    merged_model_dir: str,
    torch_dtype=torch.bfloat16,
    device_map={"": "cpu"}
):
    print("Loading base model and tokenizer...")
    model_base = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        load_in_8bit=False,
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    model_base_original = copy.deepcopy(model_base)

    print("Loading LoRA configuration...")
    peft_config = PeftConfig.from_pretrained(lora_model_dir)

    print("Loading LoRA model and applying weights...")
    model_lora = PeftModel.from_pretrained(
        model_base,
        lora_model_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        is_trainable=True,
        inference_mode=False # 明确禁用推理模式，保证LoRA激活
    )
    # print("LoRA modules:", model_lora.peft_config)
    # print("Trainable weights in LoRA:", [n for n, p in model_lora.named_parameters() if p.requires_grad])


    print("Merging LoRA weights into base model...")
    model_merged = model_lora.merge_and_unload()



    issame= compare_model_weights(model_base_original, model_merged)
    print(f"Model comparison result: {issame}")

    if not issame:
        print("✅ Merging is valid.")
    else:
        print("⚠️ Merging changes no params. Might be invalid.")

    print(f"Saving merged model to {merged_model_dir}...")
    model_merged.save_pretrained(merged_model_dir)
    tokenizer.save_pretrained(merged_model_dir)
    print("✅ Model merging complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--lora_model_dir", type=str, required=True)
    parser.add_argument("--merged_model_dir", type=str, required=True)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--device_map", type=str, default="cpu")

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    print("base_model_dir:", args.base_model_dir)
    print("lora_model_dir:", args.lora_model_dir)
    merge_lora_into_base(
        base_model_dir=args.base_model_dir,
        lora_model_dir=args.lora_model_dir,
        merged_model_dir=args.merged_model_dir,
        torch_dtype=dtype_map[args.torch_dtype],
        device_map=args.device_map
    )
