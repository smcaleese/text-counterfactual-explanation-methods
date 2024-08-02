import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer, BertForMaskedLM
from CLOSS.closs import generate_counterfactual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load models
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model_name = "textattack/bert-base-uncased-SST-2"

sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)
sentiment_model.to(device)

sentiment_model_tokenizer = AutoTokenizer.from_pretrained(model_name)

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

LM_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
LM_model.lm_head = LM_model.cls

def get_all_embeddings(model, tokenizer):
    all_word_embeddings = torch.zeros((tokenizer.vocab_size, 768)).detach().to(device)
    for i in range(tokenizer.vocab_size):
        input_tensor = torch.tensor(i).view(1, 1).to(device)
        word_embedding = model.bert.embeddings.word_embeddings(input_tensor)
        all_word_embeddings[i, :] = word_embedding
    all_word_embeddings = all_word_embeddings.detach().requires_grad_(False)
    return all_word_embeddings

all_word_embeddings = get_all_embeddings(sentiment_model, sentiment_model_tokenizer)

def calculate_score(text, sentiment_model_tokenizer, dataset, device):
    def tokenize_with_correct_token_type_ids(input_text, tokenizer):
        # Tokenize the input
        tokens = tokenizer(input_text, return_tensors="pt", padding=True)
        
        # Get the position of the first [SEP] token
        sep_pos = (tokens.input_ids == tokenizer.sep_token_id).nonzero()[0, 1].item()
        
        # Create token_type_ids
        token_type_ids = torch.zeros_like(tokens.input_ids)
        token_type_ids[0, sep_pos+1:] = 1  # Set to 1 after the first [SEP] token
        
        # Update the tokens dictionary
        tokens['token_type_ids'] = token_type_ids
        
        return tokens

    if type(text) == list:
        if type(text[0]) == str:
            tokens = text
            ids = sentiment_model_tokenizer.convert_tokens_to_ids(tokens)
            text = sentiment_model_tokenizer.decode(ids[1:-1])
        elif type(text[0]) == int:
            ids = text
            text = sentiment_model_tokenizer.decode(ids[1:-1])
            print(f"text: {text}")

    if dataset == "sst_2":
        inputs = sentiment_model_tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
    elif dataset == "qnli":
        inputs = tokenize_with_correct_token_type_ids(text, sentiment_model_tokenizer).to(device)

    logits = sentiment_model(**inputs).logits
    prob_positive = torch.nn.functional.softmax(logits, dim=1)[0][1].item()
    return prob_positive


def main():
    text = "I thought the movie was terrible and one of the worst I've ever seen."
    # text = "I thougth this movie was amazing and one of the best I've ever seen."

    id_list = sentiment_model_tokenizer.encode(text, add_special_tokens=True, truncation=True)
    print(id_list, id_list[0], type(id_list), type(id_list[0]))
    # tokens = sentiment_model_tokenizer.convert_ids_to_tokens(id_list)
    # print(f"Tokens: {tokens}, {type(tokens)}")

    score = calculate_score(id_list, sentiment_model_tokenizer, "sst_2", device)
    print(f"Score: {score}")

if __name__ == "__main__":
    main()
