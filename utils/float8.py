import torch

def module_filter_fn(mod, fqn):
    if isinstance(mod, torch.nn.Linear):
        if any(name in fqn for name in [
            "expert", "qkv_proj", "out_proj", "up_proj", "gate_proj",
            "down_proj", "q_proj", "kv_proj"
        ]):
            return True
    return False
