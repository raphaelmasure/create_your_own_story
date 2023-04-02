
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token='<PAD>', padding_side='right', return_attention_mask=True)

special_tokens_dict = {
    "bos_token": "<BOS>",
    "eos_token": "<EOS>",
    "pad_token": "<PAD>",
    }
tokenizer.add_special_tokens(special_tokens_dict)

# Load the trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load('./models/model_d1.pth', map_location='cpu'))
model.eval()


torch.save(model, 'model_d1.pth')
