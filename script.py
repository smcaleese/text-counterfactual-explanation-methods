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

    if dataset == "sst_2":
        inputs = sentiment_model_tokenizer(text, max_length=512, truncation=True, return_tensors="pt").to(device)
    elif dataset == "qnli":
        inputs = tokenize_with_correct_token_type_ids(text, sentiment_model_tokenizer).to(device)

    logits = sentiment_model(**inputs).logits
    prob_positive = torch.nn.functional.softmax(logits, dim=1)[0][1].item()
    return prob_positive

def generate_closs_counterfactual(text, args):
    counterfactual_text = generate_counterfactual(
        text,
        sentiment_model,
        LM_model,
        calculate_score,
        sentiment_model_tokenizer,
        all_word_embeddings,
        device,
        args
    )
    return counterfactual_text

def main():
    df_input = pd.read_csv(f"input/sst-input.csv")
    print(f"df_input size: {df_input.shape}")

    # args = {
    #     "beam_width": 15,
    #     "w": 5,
    #     "K": 30,
    #     "tree_depth": 0.3,
    #     "substitution_evaluation_method": "hotflip_only",
    #     "substitution_gen_method": "hotflip_only",
    #     "dataset": "sst_2"
    # }

    args = {
        "beam_width": 15,
        "w": 5,
        "K": 30,
        "tree_depth": 0.9,
        "substitutions_after_loc": 0.15,
        # "substitution_evaluation_method": "SVs",
        "substitution_evaluation_method": "grad_only",
        "substitution_gen_method": "no_opt_lmh",
        "dataset": "sst_2"
    }

    text = "I really loved the movie because of the great acting and the amazing plot."

    print(f"Text: {text}")
    counterfactual_text = generate_closs_counterfactual(text, args)
    print(f"Counterfactual Text: {counterfactual_text}")

if __name__ == "__main__":
    main()
