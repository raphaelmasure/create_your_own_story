
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer

from torch.utils.data import Dataset,  random_split



# Download the pre-trained GPT-Neo model's tokenizer
# Add the custom tokens denoting the beginning and the end
# of the sequence and a special token for padding
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium",
#                             bos_token="<BOS>",
#                             eos_token="<EOS>",
#
#
#                         pad_token="<PAD>")
#Get the tokenizer and model
print('load tokenizer')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model.resize_token_embeddings(len(tokenizer))
# model.load_state_dict(torch.load('models/GPT2-med-2048-512.pt', map_location=torch.device('cpu')))
# model.load_state_dict(torch.load('models/GPT2-small.pt'))

# print('add special tokens')
special_tokens_dict = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
    }
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print('initialize model')

# Load the saved model
"""
model = GPT2Model.from_pretrained('gpt2')
model.load_state_dict(torch.load('gpt2_model.pth', map_location=torch.device('cpu')))"""
#load the model
model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()

# model.resize_token_embeddings(len(tokenizer))
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# Download the pre-trained GPT-Neo model and transfer it to the GPU
# print('initilize model')

# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").cuda()
# print('resize model')

# Resize the token embeddings because we've just added 3 new tokens
model.resize_token_embeddings(len(tokenizer))

with open('../../processed_data/textes.txt', "r", encoding='utf-8-sig') as file:
    data = file.readlines()
print(len(data))
class NetflixDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            # Encode the descriptions using the GPT-Neo tokenizer
            encodings_dict = tokenizer(txt,
                                        truncation='longest_first',
                                        max_length=max_length,
                                        padding="max_length")
            input_ids = torch.tensor(encodings_dict['input_ids'])
            self.input_ids.append(input_ids)
            mask = torch.tensor(encodings_dict['attention_mask'])
            self.attn_masks.append(mask)

    def __len__(self):
     return len(self.input_ids)

    def __getitem__(self, idx):
     return self.input_ids[idx], self.attn_masks[idx]

dataset = NetflixDataset(data, tokenizer, 512)

train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset,
                            [train_size, len(dataset) - train_size])

#Params list with explications
"""output_dir="models/gpt-small",
num_train_epochs=1,
logging_steps= Number of update steps between two logs if logging_strategy="steps".
save_steps= Number of updates steps before two checkpoint saves if save_strategy="steps".
per_device_train_batch_size = The batch size per GPU/TPU core/CPU for training.
per_device_eval_batch_size = The batch size per GPU/TPU core/CPU for evaluation.
warmup_steps = Number of steps used for a linear warmup from 0 to learning_rate. Overrides any effect of warmup_ratio.
weight_decay = The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.
logging_dir = TensorBoard log directory. Will default to *output_dir/runs/CURRENT_DATETIME_HOSTNAME*."""

training_args = TrainingArguments(output_dir="models/gpt-small",
                                  num_train_epochs=1,
                                  logging_steps=1000,
                                  save_steps=1000,
                                  per_device_train_batch_size=2,
                                  per_device_eval_batch_size=2,
                                  warmup_steps=100,
                                  weight_decay=0.01,
                                  logging_dir="logs/")

trainer = Trainer(model=model, args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=val_dataset,
                  # This custom collate function is necessary
                  # to built batches of data
                  data_collator=lambda data:
              {"input_ids": torch.stack([f[0] for f in data]),
               "attention_mask": torch.stack([f[1] for f in data]),
               "labels": torch.stack([f[0] for f in data])})
# Start training process!
print('start training')
trainer.train()

#Save the model
torch.save(model.state_dict(), 'model_beg.pth')

tokenizer.save_pretrained('tokenizer/gpt-small')


# Set the pad token id
model.config.pad_token_id = tokenizer.eos_token_id

"""#List to store the story
output = []

#Generate text with an input
generated = tokenizer.encode('It was a cold winter, with shy suns and heavy storms', return_tensors="pt").cuda()
sample_outputs = model.generate(generated, do_sample=False, top_k=50, max_length=1024, top_p=0.95,
        num_return_sequences=1, repetition_penalty=1.5).cuda()

predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
#Store the story in a list
output.append(predicted_text)


#Chain the generation
for d in range(0,2):

    cutter=int(len(output[d])*(0.5))
    #Generate text with the end of the previous text
    input_seq = output[d][cutter:]
    #if no text was generated, let's stop
    if input_seq.strip() == "":
        break

    # Generate text with the end of the previous text
    input_ids = tokenizer.encode(input_seq, return_tensors='pt').cuda()
    # Generate the next sequence
    sample_outputs = model.generate(input_ids=input_ids, do_sample=False, temperature=0.6, max_length=1024, top_k=50, top_p=0.95, repetition_penalty=1.5)
    # Decode the generated sequence
    generated_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

    #Clean the story : only keep the generated part
    only_the_generated_part=generated_text[len(input_seq):]
    #Store the story in a list
    output.append(only_the_generated_part)

#Print the result
for out in output:
    print(out + '\n')"""
