
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import gensim
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def lemmatize_stemming(text):
    stemmer = PorterStemmer()
    text = stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='n'))
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and token not in stopwords.words('english') and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result
"""
#Tokenizer adapted with our specials tokens
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token='<PAD>', padding_side='right', return_attention_mask=True)

special_tokens_dict = {
    "bos_token": "<BOS>",
    "eos_token": "<EOS>",
    "pad_token": "<PAD>",
    "additional_special_tokens": ["<fairy>", ]
    }
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# Load the trained model

model = GPT2LMHeadModel.from_pretrained('gpt2')
model_path = torch.load('gpt2_model2.pth', map_location=device) #Path of the model
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(model_path, strict=False)
model.eval()
"""
def init_generate(initial_text, model, tokenizer, max_length):
    """initial_text : Take an input text to initiate the generation (str)
    model : pretrained model (must be a model)
    tokenizer : pretrained model (must be a tokenizer)
    max_length : number of text generated(int)
    """
    
    # Set the pad token id
    model.config.pad_token_id = tokenizer.eos_token_id
    
    #Generate text with an input
    generated = tokenizer.encode(initial_text, return_tensors="pt")
    sample_outputs = model.generate(generated, do_sample=False, temperature=0.8, top_k=50, max_length=max_length, top_p=0.95,
            num_return_sequences=1, repetition_penalty=1.5)

    return tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

 


def generate(initial_text, model, tokenizer, number_of_choices=2, max_length=300):
    """initial_text : Take an input text to initiate the generation (str)
    model : pretrained model (must be a model)
    tokenizer : pretrained model (must be a tokenizer)
    number_of_choices : number of senetences returned (int)
    max_length : number of text generated (int)
    """
    # Set the pad token id
    model.config.pad_token_id = tokenizer.eos_token_id
    
    #Generate text with an input
    generated = tokenizer.encode(initial_text, return_tensors="pt")
    sample_outputs = model.generate(generated, do_sample=True, temperature=0.8, top_k=50, max_length=max_length, top_p=0.95,
            num_return_sequences=number_of_choices, repetition_penalty=1.5)
    output=[]
    for i, sample_output in enumerate(sample_outputs):
        output.append("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True))[len(initial_text):])
    return output



def final_generate(initial_text, model, tokenizer,  max_length=200):
    """initial_text : Take an input text to initiate the generation (str)
    model : pretrained model (must be a model)
    tokenizer : pretrained model (must be a tokenizer)
    number_of_choices : number of senetences returned (int)
    max_length : number of text generated (int)
    """
    # Set the pad token id
    model.config.pad_token_id = tokenizer.eos_token_id
    

    #Generate text with an input
    generated = tokenizer.encode(initial_text, return_tensors="pt")
    sample_outputs = model.generate(generated, do_sample=False, temperature=0.8, top_k=50, max_length=max_length, top_p=0.95,
            num_return_sequences=1, repetition_penalty=1.5)

    return tokenizer.decode(sample_outputs[0], skip_special_tokens=True)[len(initial_text):]


def print(outputs):
    """outputs (list of str) : take a list of text and print them """
    #Print the result
    for out in outputs:
        print(out + '\n')
