import torch
import math
from tqdm import tqdm

def compute_perplexity(model, dataset, device="cuda"):
    model.eval()
    losses = []
    for batch in tqdm(dataset, desc="Calculando PPL"):
        input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)
