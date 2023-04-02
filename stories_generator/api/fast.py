import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from stories_generator.ml_logic.generator import init_generate, generate, final_generate, preprocess
from stories_generator.ml_logic.model import load_body, load_end, load_beg

app = FastAPI()
app.state.tokenizer_body, app.state.model_body = load_body()
app.state.tokenizer_beg, app.state.model_beg = load_beg()
app.state.tokenizer_end, app.state.model_end = load_end()

# Optional, good practice for dev purposes. Allow all middlewares
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
#Première génération de texte
@app.get("/init_generate")
def first_predict(initial_text="it was a small town boy", tokenizer=app.state.tokenizer_beg,  model=app.state.model_beg, max_length=400):
    """Generate a text through
    initial_text (str) : Take an input text to initiate the generation
        model : pretrained model (must be a model)
        tokenizer : pretrained model (must be a tokenizer)
        max_length (int) = number of text generated
        """
    initial_text2= ' '.join(preprocess(initial_text[:-100])) + initial_text[-100:]
    if len(initial_text)+400<1024:
        pass
    else :
        initial_text=initial_text2[-960:]
        max_length=1024
    return {0: f'Once upon a time {init_generate(initial_text, model, tokenizer, max_length=max_length)}'}

#génération de texte
@app.get("/generate")
def predict(initial_text, tokenizer=app.state.tokenizer_body,  model=app.state.model_body, max_length=200):
    """Generate a text through
    initial_text (str) : Take an input text to initiate the generation
        model : pretrained model (must be a model)
        tokenizer : pretrained model (must be a tokenizer)
        max_length (int) = number of text generated
        """
    initial_text2= ' '.join(preprocess(initial_text[:-100])) + initial_text[-100:]
    if len(initial_text2)+30<1024:
        max_length=len(initial_text2)+30
        initial_text=initial_text2
    else :
        initial_text=initial_text2[-990:]
        max_length=1024
    return {0: generate(initial_text, model, tokenizer, max_length=max_length)}

#Dernière génération de texte
@app.get("/final_generate")
def last_predict(initial_text, tokenizer=app.state.tokenizer_end,  model=app.state.model_end, max_length=200):
    """Generate a text through
    initial_text (str) : Take an input text to initiate the generation
        model : pretrained model (must be a model)
        tokenizer : pretrained model (must be a tokenizer)
        max_length (int) = number of text generated
        """
    initial_text2= ' '.join(preprocess(initial_text[:-100])) + initial_text[-100:]
    if (len(initial_text2)+200)<1024:
        max_length=len(initial_text2)+200
        initial_text=initial_text2
    else :
        initial_text=initial_text2[-823:]
        max_length=1024
    return {0: final_generate(initial_text, model, tokenizer, max_length=max_length)}


@app.get("/")
def root():
    return {
    'greeting': 'Hello'
}
