from transformers import  GPT2Tokenizer
import torch

def load_body():
#Tokenizer adapted with our specials tokens
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token='<PAD>', padding_side='right', return_attention_mask=True)

    special_tokens_dict = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
        }
    tokenizer.add_special_tokens(special_tokens_dict)

    # Load the trained model

    model = torch.load('stories_generator/ml_logic/models/model_guillaume.pth', map_location=torch.device('cpu'))
    return tokenizer , model

def load_beg():
#Tokenizer adapted with our specials tokens
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token='<PAD>', padding_side='right', return_attention_mask=True)

    special_tokens_dict = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
        }
    tokenizer.add_special_tokens(special_tokens_dict)

    # Load the trained model

    model = torch.load('stories_generator/ml_logic/models/model_beg.pth', map_location=torch.device('cpu'))
    return tokenizer , model

def load_end():
#Tokenizer adapted with our specials tokens
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token='<PAD>', padding_side='right', return_attention_mask=True)

    special_tokens_dict = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
        }
    tokenizer.add_special_tokens(special_tokens_dict)

    # Load the trained model

    model = torch.load('stories_generator/ml_logic/models/end_model.pth', map_location=torch.device('cpu'))
    return tokenizer , model
