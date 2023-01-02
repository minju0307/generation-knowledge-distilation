import argparse
import json

import torch
import pandas as pd
import numpy as np
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rich.table import Column, Table
from rich import box
from rich.console import Console

from utils import *
from metric import calculateF1, split
from train import YourDataSetClass, display_df


def validate(epoch, tokenizer, model, device, loader):
    """
    Function to evaluate model for predictions

    """

    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            if _ % 10 == 0:
                console.print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default='cins')
    parser.add_argument("--output_dir", type=str, default='outputs')
    parser.add_argument("--target_type", type=str, default='target')
    parser.add_argument("--ner_position", type=str, default="input")
    parser.add_argument("--example_type", type=str, default='easy')

    args = parser.parse_args()

    console = Console(record=True)
    device = 'cuda' if cuda.is_available() else 'cpu'

    model_params = {
        "MODEL": f"./{args.output_dir}/best_model",  # model_type: kt-ulm-base
        "TRAIN_BATCH_SIZE": 8,  # training batch size
        "VALID_BATCH_SIZE": 8,  # validation batch size
        "TRAIN_EPOCHS": 5,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 1e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 256 if args.prompt == 'driven' else 768 if args.example_type == 'hard' else 512,
        # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 256,  # max length of target text
        "SEED": 42  # set seed for reproducibility

    }
    test_params = {
        'batch_size': model_params["VALID_BATCH_SIZE"],
        'shuffle': False,
        'num_workers': 0
    }

    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    with open('./data_learn/klue_ner_test_20.txt', 'r', encoding='utf-8') as f:
        test = f.readlines()

    test = preprocessing(test)
    if args.prompt == 'example':
        with open(f"./{args.output_dir}/" + "example.json", 'r') as f:
            example = json.load(f)
        pm = PromptMaker(example=example, args=args)
    else:
        pm = PromptMaker(args=args)

    test_df = pm.get_df(args.prompt, test)
    display_df(test_df.head(2))
    console.log(f"{'-' * 20}input text example{'-' * 20}\n{test_df['prompt'][0]}")

    test_set = YourDataSetClass(test_df, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"],
                                model_params["MAX_TARGET_TEXT_LENGTH"], 'prompt', args.target_type)
    test_loader = DataLoader(test_set, **test_params)
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(epoch, tokenizer, model, device, test_loader)
        final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
    display_df(final_df.head(2))
    pattern = '<[^<>]+:[^<>]+>'
    pred = final_df['Generated Text'].apply(lambda x: re.findall(pattern, x.replace('<pad>', ''))).values.tolist()
    label = final_df['Actual Text'].apply(lambda x: re.findall(pattern, x.replace('<pad>', ''))).values.tolist()

    pred2 = [list(map(split, i)) for i in pred]
    label2 = [list(map(split, i)) for i in label]

    predict_golden = zip(pred2, label2)
    p, r, f1 = calculateF1(predict_golden)

    print(f"precision:{p}\nrecall:{r}\nf1:{f1}")
    with open(f"./{args.output_dir}/test_result.txt", "w") as f:
        f.write(f"precision:{p}\nrecall:{r}\nf1:{f1}")

