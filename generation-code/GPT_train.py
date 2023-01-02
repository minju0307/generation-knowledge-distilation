import pandas as pd
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# Importing the T5 modules from huggingface/transformers
from transformers import AutoTokenizer, GPT2LMHeadModel, GPTJForCausalLM

from rich.table import Column, Table
from rich import box
from rich.console import Console

import argparse


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

    source = self.tokenizer.batch_encode_plus([source_text], max_length= self.source_len, pad_to_max_length=True, truncation=True, return_tensors='pt')
    target = self.tokenizer.batch_encode_plus([target_text], max_length= self.target_len,  pad_to_max_length=True, truncation=True, return_tensors='pt')

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


def train(epoch, tokenizer, model, device, loader, optimizer):

  """
  Function to be called for training with the parameters passed from main function

  """

  model.train()
  for _,data in enumerate(loader, 0):
    ids = data['source_ids'].to(device, dtype = torch.long)
    mask = data['source_mask'].to(device, dtype = torch.long)
    lm_labels=ids

    outputs = model(input_ids = ids, attention_mask = mask, labels=lm_labels)
    loss = outputs[0]

    if _%100==0:
      training_logger.add_row(str(epoch), str(_), str(loss))
      console.print(training_logger)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


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
          #print([tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in ids])
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask,
              max_length=1024,
              num_beams=2,
              repetition_penalty=10.0,
              length_penalty=10.0,
              early_stopping=True
              )

          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          print(preds)
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
          if _%100==0:
              console.print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals


def GPTTrainer(dataframe, test_dataframe, source_text, target_text, model_params, output_dir="./GPT_outputs/" ):

    """
    GPT trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"]) # pytorch random seed
    np.random.seed(model_params["SEED"]) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = AutoTokenizer.from_pretrained(model_params["MODEL"], padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    #model = GPTJForCausalLM.from_pretrained(model_params["MODEL"], revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = GPT2LMHeadModel.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text,target_text]]
    test_dataframe = test_dataframe[[source_text, target_text]]
    display_df(dataframe.head(2))
    display_df(test_dataframe.head(2))


    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    # train_size = 0.8
    # train_dataset=dataframe.sample(frac=train_size,random_state = model_params["SEED"])
    # val_dataset=dataframe.drop(train_dataset.index).reset_index(drop=True)
    # train_dataset = train_dataset.reset_index(drop=True)

    train_dataset=dataframe
    val_dataset=test_dataframe

    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")


    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    #print(training_set['source_ids'].shape)
    #print(training_set['target_ids'].shape)
    val_set = YourDataSetClass(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)


    # Defining the parameters for creation of dataloaders
    train_params = {
      'batch_size': model_params["TRAIN_BATCH_SIZE"],
      'shuffle': True,
      'num_workers': 0
      }


    val_params = {
      'batch_size': model_params["VALID_BATCH_SIZE"],
      'shuffle': False,
      'num_workers': 0
      }


    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=model_params["LEARNING_RATE"])


    # Training loop
    console.log(f'[Initiating Fine Tuning]...\n')

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")
    predictions, actuals = validate(tokenizer, model, device, val_loader)
    final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
    final_df.to_csv(os.path.join(output_dir, f'predictions.csv'))

    console.log(f"[TEST Completed.]\n")

    console.log(f"[Saving Model]...\n")
    #Saving the model after training
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


    # evaluating test dataset


    console.save_text(os.path.join(output_dir,'logs.txt'))

    console.log(f"[Validation Completed.]\n")
    console.print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")
    console.print(f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n""")
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")


if __name__=='__main__':
    # define a rich console logger
    console = Console(record=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--few_shot", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=3)
    args = parser.parse_args()

    #load dataset
    df = pd.read_csv("data/ROCStories__spring2016.csv")
    # df = df[:50]
    # df["story"] = df["sentence1"] + ' ' + df["sentence2"] + ' ' + df[
    #     "sentence3"] + ' ' + df["sentence4"]+' ' + df["sentence5"]+' '+'<|endoftext|>'
    # df["ending"] = df["sentence5"]
    data_num=10000
    df = df[:data_num*args.few_shot]
    data_num=data_num*args.few_shot
    #print(data_num)

    prompts=[]
    endings=[]
    for i in range(0,data_num, args.few_shot):
        #prompt="Write the ending of the following story\n"
        prompt=''

        for j in range(i, i+args.few_shot):
            story=df["sentence1"][j] + ' ' + df["sentence2"][j] + ' ' + df["sentence3"][j] + ' ' + df["sentence4"][j]
            ending = df["sentence5"][j]
            prompt= prompt+'\n story: '+story+' ending: '+ending+'<|endoftext|>\n'
        prompts.append(prompt)
        endings.append(ending)

    #print(prompts)
    #print(endings)

    df = pd.DataFrame({'story': prompts, 'ending': endings})

    #load test dataset
    df2 = pd.read_csv("data/ROCStories_winter2017.csv")
    # df2 = df2[:10]
    # df2["story"] = df2["sentence1"] + ' ' + df2["sentence2"] + ' ' + df2[
    #     "sentence3"] + ' ' + df2["sentence4"]
    # df2["ending"] = df2["sentence1"] + ' ' + df2["sentence2"] + ' ' + df2[
    #     "sentence3"] + ' ' + df2["sentence4"]+' '+df2["sentence5"]

    data_num=10000
    df2 = df2[:data_num*args.few_shot]
    data_num=data_num*args.few_shot
    #print(data_num)

    prompts=[]
    endings=[]
    for i in range(0, data_num, args.few_shot ):
        #prompt="Write the ending of the following story\n"
        #prompt = "Write the right ending of the following story.\n"\
                 # +" The right ending should not contradict the story but should be something that will happen after the story.\n" \
                 # + " The right ending should be just one sentence.\n"
        prompt=''

        for j in range(i, i+args.few_shot):
            story=df2["sentence1"][j] + ' ' + df2["sentence2"][j] + ' ' + df2["sentence3"][j] + ' ' + df2["sentence4"][j]
            ending = df2["sentence5"][j]
            if not j == (i + args.few_shot - 1):
                prompt= prompt+'\n story: '+story+' ending: '+ending+'<|endoftext|>\n'
            else:
                prompt = prompt + '\n story: ' + story + ' ending:\n'

        prompts.append(prompt)
        endings.append(ending)

    #print(prompts)
    #print(endings)
    #print(len(prompts))
    #print(len(endings))
    df2 = pd.DataFrame({'story': prompts, 'ending': endings})


    #training logger
    training_logger = Table(Column("Epoch", justify="center"),
                            Column("Steps", justify="center"),
                            Column("Loss", justify="center"),
                            title="Training Status", pad_edge=False, box=box.ASCII)

    #GPU or CPU
    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'

    #model_params
    # "MODEL": "EleutherAI/gpt-j-6B"
    model_params = {
        "MODEL": "gpt2",
        "TRAIN_BATCH_SIZE": 8,  # training batch size
        "VALID_BATCH_SIZE": 8,  # validation batch size
        "TRAIN_EPOCHS": args.epoch,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 1e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 512,  # max length of target text
        "SEED": 42  # set seed for reproducibility
    }

    output_dir=f'GPT2_outputs_endoftext_1shot_epoch{args.epoch}_data10000_10000'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)
    GPTTrainer(dataframe=df, test_dataframe=df2, source_text="story", target_text="ending", model_params=model_params, output_dir=output_dir)