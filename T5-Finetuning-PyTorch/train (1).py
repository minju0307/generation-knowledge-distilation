from dataclass import YourDataSetClass
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
from torch import cuda
# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration,T5Config
import logging

# rich: for a better display on terminal
#from rich.table import Column, Table
#from rich import box
#from rich.console import Console


# Setting up the device for GPU usage

device = 'cuda' if cuda.is_available() else 'cpu'
logger = logging.getLogger(__name__)

model_params = {
    "MODEL": "KETI-AIR/ke-t5-base",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 16,  # training batch size
    "VALID_BATCH_SIZE": 8,  # validation batch size
    "TRAIN_EPOCHS": 300,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 128,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 128,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}

def train(epoch, tokenizer, model, device, loader, optimizer):

    """
    Function to be called for training with the parameters passed from main function

    """

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 10 == 0:
            #training_logger.add_row(str(epoch), str(_), str(loss))
            print(str(epoch)+str(' : ')+str(_)+str(loss))
            logger.info(str(epoch)+str(_)+str(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(epoch, tokenizer, model, device, loader):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask,
              max_length=150,
              num_beams=2,
              repetition_penalty=2.5,
              length_penalty=1.0,
              early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=False, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=False, clean_up_tokenization_spaces=True)for t in y]
          if _%10==0:
              logger.info(f'Completed {_}')
              print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals

def T5Trainer(
    dataframe, source_text, target_text, model_params, output_dir="./outputs/"
):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    #console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    #config=T5Config.from_pretrained('./cofig/config.json')
    model=T5ForConditionalGeneration(T5Config(d_ff= 2048, d_kv= 64, d_model= 768, decoder_start_token_id= tokenizer.pad_token_id,
  dense_act_fn= "gelu_new",  dropout_rate= 0.1,  eos_token_id= 1,  feed_forward_proj= 'gated-gelu',  initializer_factor= 1.0,
  is_encoder_decoder= True,  is_gated_act= True,  layer_norm_epsilon= 1e-06,  model_type= "t5",  n_positions= 256,
  num_decoder_layers=12,  num_heads= 12,  num_layers= 12,  pad_token_id= 0, relative_attention_max_distance= 128,  relative_attention_num_buckets= 32,
  torch_dtype="float32",  transformers_version="4.21.1",  use_cache= True,  vocab_size= 64128))

    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    print(f"[Data]: Reading data...\n")
    logger.info(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text, target_text]]
    #display_df(dataframe.head(2))

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_size = 0.8
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print(f"FULL Dataset: {dataframe.shape}")
    print(f"TRAIN Dataset: {train_dataset.shape}")
    print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = YourDataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=model_params["LEARNING_RATE"]
    )

    # Training loop
    print(f"[Initiating Fine Tuning]...\n")
    logger.info(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)
    print(f"[Saving Model]...\n")
    logger.info(f"[Saving Model]...\n")
    # Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

    # evaluating test dataset
    logger.info(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):
        predictions, actuals = validate(epoch, tokenizer, model, device, val_loader)
        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    #console.save_text(os.path.join(output_dir, "logs.txt"))

    logger.info(f"[Validation Completed.]\n")
    logger.info(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    logger.info(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    logger.info(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--data_path", type=str, required=True)
    cli_args = cli_parser.parse_args()
    path = cli_args.data_path
    df = pd.read_csv(path,encoding='utf-8-sig')
    print("go model")
    T5Trainer(
        dataframe=df,
        source_text="original",
        target_text="target",
        model_params=model_params,
        output_dir="outputs",
    )