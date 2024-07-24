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

def generate_closs_counterfactual(text, args):
    counterfactual_text = generate_counterfactual(
        text,
        sentiment_model,
        LM_model,
        sentiment_model_tokenizer,
        all_word_embeddings,
        device,
        args
    )
    return counterfactual_text

def main():
    df_input = pd.read_csv(f"input/sst-input.csv")
    # text = df_input.iloc[0]["original_text"]
    # text = "I really loved the movie."
    text = "I thought the movie was terrible and one of the worst I've ever seen."
    print(f"Original text: {text}")

    args = {
        "beam_width": 15,
        "w": 5,
        "K": 30,
        "substitution_evaluation_method": "hotflip_only",
        "substitution_gen_method": "hotflip_only"
    }

    # args = {
    #     "beam_width": 15,
    #     "w": 5,
    #     "K": 30,
    #     "substitution_evaluation_method": "SVs",
    #     "substitution_gen_method": "no_opt_lmh"
    # }

    counterfactual_text = generate_closs_counterfactual(text, args)

if __name__ == "__main__":
    main()
