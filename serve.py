import torch
import flask
import joblib
import functools
import time
from flask import Flask
from flask import request
from transformers import BertForMaskedLM, BertTokenizer

app = Flask(__name__)

DEVICE = "cpu"
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
PREDICTION_DICT = dict()
def mask_prediction(sentence, mask):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')                                                                                               
    mask_index = int(mask) # are                                                                                                                                         
    text = str(sentence) #"Hello how [MASK] you doing?"                                                                                                          
    tokenized_text = tokenizer.tokenize(text)  
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])    
    with torch.no_grad():
        out, = model(tokens_tensor)
        output = out[0]
    predicted_index = torch.argmax(output, dim=1)[mask_index].item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    return predicted_token                                                                                                                                        
    print(predicted_token)

@app.route("/predict")
def predict():
    sentence = request.args.get("sentence")
    mask_index  = request.args.get('mask_index')
    start_time = time.time()
    prediction = mask_prediction(sentence,mask_index)
    response = {}
    response["response"] = {
        'mask': str(prediction),
        'sentence': str(sentence),
        'mask_index': str(mask_index),
        'time_taken': str(time.time() - start_time)
    }
    return flask.jsonify(response)

if __name__ == "__main__":
    model.to(DEVICE)
    model.eval()
    app.run()
