from fastapi import FastAPI, Response, status

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification

app = FastAPI()

@app.get("/word/{sentence}", status_code=200)
async def get_suggested_word(sentence: str, response: Response):

    if not sentence:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"response": "The prompt needs to be a string"}
    if "<blank>" not in sentence:
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"response": "The prompt need to contain the keyword <blank>"}

    suggestion_model_name = "bert-base-uncased"
    suggestion_tokenizer = AutoTokenizer.from_pretrained(suggestion_model_name)
    suggestion_model = AutoModelForMaskedLM.from_pretrained(suggestion_model_name)

    sentence = sentence.replace("<blank>", "[MASK]")
    input_ids = suggestion_tokenizer.encode(sentence, return_tensors="pt")
    mask_index = torch.where(input_ids == suggestion_tokenizer.mask_token_id)[1]

    with torch.no_grad():
        output = suggestion_model(input_ids)
        logits = output[0]

    softmax_logits = torch.softmax(logits[0, mask_index], dim=-1)
    top_indices = torch.topk(softmax_logits, k=3, dim=-1).indices

    suggestions = []
    for index in top_indices[0]:
        suggestion = suggestion_tokenizer.decode(index)
        suggestions.append(suggestion)

    sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

    def analyze_sentiment(suggestions):
        encoded_input = sentiment_tokenizer(suggestions, return_tensors='pt')
        with torch.no_grad():
            output = sentiment_model(**encoded_input)
        logits = output.logits
        predicted_labels = torch.argmax(logits, dim=1)
        return predicted_labels

    sentiment_scores = analyze_sentiment(suggestions)

    response.status_code = status.HTTP_200_OK
    return {"message": f"{suggestions, sentiment_scores}"}
