from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name, device="cuda", use_8bit=False):
    kwargs = {"device_map": "auto", "torch_dtype": "auto"}
    if use_8bit:
        kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer
