import pandas as pd
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

from rich.table import Column, Table
from rich import box
from rich.console import Console

def display_df(df):
  """display dataframe in ASCII format"""

  console=Console()
  table = Table(Column("source_text", justify="center" ), Column("target_text", justify="center"), title="Sample Data",pad_edge=False, box=box.ASCII)

  for i, row in enumerate(df.values.tolist()):
    table.add_row(row[0], row[1])

  console.print(table)

class YourDataSetClass(Dataset):
  """
  Creating a custom dataset for reading the dataset and
  loading it into the dataloader to pass it to the neural network for finetuning the model

  """

  def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
    self.tokenizer = tokenizer
    self.data = dataframe
    self.source_len = source_len
    self.target_len = target_len
    self.target_text = self.data[target_text]
    self.source_text = self.data[source_text]

  def __len__(self):
    return len(self.target_text)

  def __getitem__(self, index):
    source_text = str(self.source_text[index])
    target_text = str(self.target_text[index])

    #cleaning data so as to ensure data is in string type
    source_text = ' '.join(source_text.split())
    target_text = ' '.join(target_text.split())

    source = self.tokenizer.batch_encode_plus([source_text], max_length= self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
    target = self.tokenizer.batch_encode_plus([target_text], max_length= self.target_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

    source_ids = source['input_ids'].squeeze()
    source_mask = source['attention_mask'].squeeze()
    target_ids = target['input_ids'].squeeze()
    target_mask = target['attention_mask'].squeeze()

    return {
        'source_ids': source_ids.to(dtype=torch.long),
        'source_mask': source_mask.to(dtype=torch.long),
        'target_ids': target_ids.to(dtype=torch.long),
        'target_ids_y': target_ids.to(dtype=torch.long)
    }

def validate(tokenizer, model, device, loader):

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
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          if _%100==0:
              console.print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals

def T5Test(dataframe, source_text, target_text, model_params, output_dir="./outputs/" ):

    """
    T5 test

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"]) # pytorch random seed
    np.random.seed(model_params["SEED"]) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text,target_text]]
    display_df(dataframe.head(2))


    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    test_dataset=dataframe

    console.print(f"TEST Dataset: {test_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    test_set = YourDataSetClass(test_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)


    # Defining the parameters for creation of dataloaders

    test_params = {
      'batch_size': model_params["VALID_BATCH_SIZE"],
      'shuffle': False,
      'num_workers': 0
      }

    # evaluating test dataset
    val_loader = DataLoader(test_set, **test_params)

    predictions, actuals = validate(tokenizer, model, device, val_loader)
    final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
    final_df.to_csv(os.path.join(output_dir, f'predictions.csv'))

    console.log(f"[TEST Completed.]\n")

    console.print(f"""[TEST] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n""")


if __name__=='__main__':
    # define a rich console logger
    console = Console(record=True)

    #load dataset
    df = pd.read_csv("data/ROCStories_winter2017.csv")
    df = df[:10000]
    df["story"] = "Write the ending of the following story: " + df["sentence1"] + ' ' + df["sentence2"] + ' ' + df[
        "sentence3"] + ' ' + df["sentence4"]
    df["ending"] = df["sentence5"]
    console.print(df['story'][0])
    console.print(df['ending'][0])

    #training logger
    training_logger = Table(Column("Epoch", justify="center"),
                            Column("Steps", justify="center"),
                            Column("Loss", justify="center"),
                            title="Training Status", pad_edge=False, box=box.ASCII)

    #GPU or CPU
    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'

    #model_params
    model_params = {
        "MODEL": "T5_outputs_2/model_files",  # model_type: t5-base/t5-large
        "VALID_BATCH_SIZE": 8,  # validation batch size
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 512,  # max length of target text
        "SEED": 42  # set seed for reproducibility
    }

    output_dir=f'test_for_{model_params["MODEL"]}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    T5Test(dataframe=df, source_text="story", target_text="ending", model_params=model_params, output_dir=output_dir)