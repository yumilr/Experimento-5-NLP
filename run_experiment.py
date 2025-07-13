from datasets.prepare_data import load_and_tokenize
from models.load_model import load_model
from evaluate.compute_perplexity import compute_perplexity

def main():
    configs = [
        {"model_name": "meta-llama/Llama-2-7b-hf", "use_8bit": False, "label": "LLaMA-2B"},
        {"model_name": "meta-llama/Llama-2-7b-hf", "use_8bit": True,  "label": "LLaMA-2B-8bit"},
        {"model_name": "BitNet/bitnet-b1.58-2B-4T-bf16", "use_8bit": False, "label": "BitNet b1.58"}
    ]

    datasets = [
        {"dataset_name": "wikitext", "config_name": "wikitext-2-raw-v1", "max_samples": 100},
        {"dataset_name": "c4", "config_name": "en", "max_samples": 50}
    ]

    for config in configs:
        print(f"\n🚀 Modelo: {config['label']}")
        model, tokenizer = load_model(config["model_name"], use_8bit=config["use_8bit"])
        
        for data in datasets:
            print(f"\n📚 Dataset: {data['dataset_name']}/{data['config_name']}")
            tokenized = load_and_tokenize(
                dataset_name=data["dataset_name"],
                config_name=data["config_name"],
                tokenizer_name=config["model_name"],
                max_samples=data["max_samples"]
            )
            ppl = compute_perplexity(model, tokenized)
            print(f"✅ Perplejidad ({config['label']} - {data['dataset_name']}): {ppl:.2f}")

if __name__ == "__main__":
    main()
