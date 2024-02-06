import torch
import numpy as np

from torchmetrics.text import CharErrorRate
from torchmetrics.text import EditDistance
from torchmetrics.text import MatchErrorRate

from src.decoders.decoders import GreedyTextDecoder
from src.io.formatting_io_ops import bcolors

def clean_special_tokens(string, tokenizer):
    for token in tokenizer.special_tokens:
        string = string.replace(token, '')

    return string


def eval_dataset(dataloader, model, dataset_name, tokenizer, wandb_session):
    decoder = GreedyTextDecoder(False)
    cer = CharErrorRate()
    ed = EditDistance()
    mer = MatchErrorRate()

    metrics = {
        f"CER_{dataset_name}": 0,
        f"ED_{dataset_name}": 0,
        f"MER_{dataset_name}": 0
    }
    total_steps = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            tokens = model(batch)['language_head_output'].cpu().detach().numpy()
            decoded_tokens = decoder({'ctc_output': tokens}, tokenizer.ctc_blank, None)

            strings = [clean_special_tokens(x, tokenizer) for x in
                       tokenizer.decode_from_numpy_list([x['text'] for x in decoded_tokens])]
            labels = [clean_special_tokens(x, tokenizer) for x in tokenizer.decode(batch['labels'].permute(1, 0))]

            metrics[f"CER_{dataset_name}"] += cer(strings, labels).item()
            metrics[f"ED_{dataset_name}"] += ed(strings, labels).item()
            metrics[f"MER_{dataset_name}"] += mer(strings, labels).item()
            total_steps += 1
        for x, y in zip(strings, labels):
            print(f"{bcolors.OKGREEN if x==y else bcolors.FAIL}Predicted:{x}, GT: {y}{bcolors.ENDC}")

    final_scores = {key: metrics[key] / total_steps for key in metrics}
    wandb_session.log(
        final_scores
    )
    return final_scores