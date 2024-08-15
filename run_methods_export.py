# %% [markdown]
# ## Setup
# 
# Mount Google Drive and clone the repository containing the methods.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import getpass

github_username = input("Enter your GitHub username: ")
github_token = getpass.getpass("Enter your GitHub personal access token: ")

# %%
repo_name = "smcaleese/masters-thesis-code"
!git clone https://{github_username}:{github_token}@github.com/{repo_name}.git

# %%
%cd masters-thesis-code
%pwd

# %% [markdown]
# Install necessary dependencies.

# %%
%pip install transformers datasets textdistance openai

# %% [markdown]
# ## Download datasets
# 
# Download the SST-2, QNLI and AG News datasets, clean the sentences, and create a list of input sentences.

# %%
num_samples = 100

# dataset = "sst_2"
dataset = "qnli"
# dataset = "ag_news"

# %%
import re

def format_sentence(sentence, dataset):
    sentence = sentence.lower()

    # remove two spaces around a comma:
    sentence = re.sub(r"\s(')\s(ve|re|s|t|ll|d)", r"\1\2", sentence)

    # remove spaces around hyphens:
    sentence = re.sub(r"-\s-", "--", sentence)
    sentence = re.sub(r"(\w)\s-\s(\w)", r"\1-\2", sentence)

    def replace(match):
        return match.group(1)

    # remove spaces before punctuation and n't:
    sentence = re.sub(r"\s([.!,?:;')]|n't)", replace, sentence)

    # remove spaces after opening parenthesis:
    sentence = re.sub(r"([(])\s", replace, sentence)

    if dataset == "qnli":
        sentence = re.sub(r"\s(\[sep\])\s", " [SEP] ", sentence)
    
    return sentence

# %%
from datasets import load_dataset

if dataset == "sst_2":
    sst = load_dataset("stanfordnlp/sst2")

    sst_sentences = sst["validation"]["sentence"]
    sst_labels = sst["validation"]["label"]

    sst_sentences_subset = sst_sentences[:num_samples]
    sst_labels_subset = sst_labels[:num_samples]

    sst_sentences_subset_formatted = [format_sentence(sentence, dataset) for sentence in sst_sentences_subset]

elif dataset == "qnli":
    qnli = load_dataset("glue", "qnli")

    qnli_questions = qnli["validation"]["question"]
    qnli_answers = qnli["validation"]["sentence"]
    qnli_labels = qnli["validation"]["label"]

    qnli_questions_subset = qnli_questions[:num_samples]
    qnli_answers_subset = qnli_answers[:num_samples]
    qnli_labels_subset = qnli_labels[:num_samples]

    qnli_questions_subset_formatted = [format_sentence(sentence, dataset) for sentence in qnli_questions_subset]
    qnli_answers_subset_formatted = [format_sentence(sentence, dataset) for sentence in qnli_answers_subset]


# %% [markdown]
# Write the sentences to a file named `sst-input.csv` and `qnli-input.csv`.

# %%
%pwd

# %%
import pandas as pd

df_sst = pd.DataFrame({
    "original_text": sst_sentences_subset_formatted,
    "original_label": sst_labels_subset
})
df_qnli = pd.DataFrame({
    "original_question": qnli_questions_subset_formatted,
    "original_answer": qnli_answers_subset_formatted,
    "original_label": qnli_labels_subset
})

df_sst.to_csv("./input/sst-input.csv", index=False)
df_qnli.to_csv("./input/qnli-input.csv", index=False)

# %% [markdown]
# ## Choose dataset

# %%
if dataset == "sst_2":
    input_file = "sst-input"
    model_name = "textattack/bert-base-uncased-SST-2"
    fizle_task = "sentiment analysis on the SST-2 dataset"
elif dataset == "qnli":
    input_file = "qnli-input"
    model_name = "textattack/bert-base-uncased-QNLI"
    fizle_task = "natural language inference on the QNLI dataset"


# %% [markdown]
# ## Create input dataframe
# 
# Columns to add to create output dataframe:
# - original_score
# - original_perplexity
# - counterfactual_text
# - counterfactual_score
# - counterfactual_perplexity
# - found_flip
# - frac_tokens_same

# %%
%pwd

# %%
import pandas as pd

df_input = pd.read_csv(f"input/{input_file}.csv")
df_input.head()

# %%
df_input.shape

# %% [markdown]
# ## Load models

# %%
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# Load the sentiment model and tokenizer.

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if dataset == "sst_2":
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    ).to(device)

elif dataset == "qnli":
    id2label = {0: "entailment", 1: "not_entailment"}
    label2id = {"entailment": 0, "not_entailment": 1}
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    ).to(device)

# elif dataset == "ag_news":
#     id2label = {
#         0: "World",
#         1: "Sports",
#         2: "Business",
#         3: "Sci/Tech"
#     }
#     label2id = {
#         "World": 0,
#         "Sports": 1,
#         "Business": 2,
#         "Sci/Tech": 3
#     }
#     sentiment_model = AutoModelForSequenceClassification.from_pretrained(
#         model_name,
#         num_labels=4,
#         id2label=id2label,
#         label2id=label2id
#     ).to(device)

sentiment_model_tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
text = "what does umc stand for? [SEP] founded in 1968 by the union of the methodist church (usa) and the evangelical united brethren church, the umc traces its roots back to the revival movement of john and charles wesley in england as well as the great awakening in the united states."
tokens = sentiment_model_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
logits = sentiment_model(**tokens).logits
prob_positive = torch.nn.functional.softmax(logits, dim=1)[0][1].item()
prob_positive

# %%
id = 1 if prob_positive > 0.5 else 0
label = id2label[id]
print(id, label)

# %% [markdown]
# Load the GPT-2 model for calculating perplexity.

# %%
from transformers import GPT2LMHeadModel, GPT2Tokenizer

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# %% [markdown]
# Load the language model for CLOSS.

# %%
import transformers

# TODO: try using a larger model to improve performance: https://arxiv.org/pdf/2111.09543
LM_model = transformers.BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
LM_model.lm_head = LM_model.cls

# %% [markdown]
# ## Helper function

# %%
import re
import textdistance

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

def calculate_perplexity(text):
    inputs = gpt2_tokenizer(text, return_tensors="pt").to(device)
    loss = gpt2_model(**inputs, labels=inputs["input_ids"]).loss
    perplexity = torch.exp(loss).item()
    return perplexity

def is_flip(original_score, counterfactual_score):
    # might need to be updated for AG News
    positive_to_negative = original_score >= 0.5 and counterfactual_score < 0.5
    negative_to_positive = original_score < 0.5 and counterfactual_score >= 0.5
    return positive_to_negative or negative_to_positive

def truncate_text(text, max_length=100):
    tokens = text.split()
    if len(tokens) > max_length:
        text = " ".join(tokens[:max_length])
    return text

def get_all_embeddings(model, tokenizer):
    all_word_embeddings = torch.zeros((tokenizer.vocab_size, 768)).detach().to(device)
    for i in range(tokenizer.vocab_size):
        input_tensor = torch.tensor(i).view(1, 1).to(device)
        word_embedding = model.bert.embeddings.word_embeddings(input_tensor)
        all_word_embeddings[i, :] = word_embedding
    all_word_embeddings = all_word_embeddings.detach().requires_grad_(False)
    return all_word_embeddings

def get_levenshtein_similarity_score(original_text, counterfactual_text):
    score = 1 - textdistance.levenshtein.normalized_distance(original_text, counterfactual_text)
    return score

# def format_polyjuice_output(text):
#     sep_token = " [SEP] "

#     # 1. handle valid texts:
#     if sep_token in text:
#         return text

#     # 2. replace invalid sep tokens (e.g. [and]):
#     pattern = re.compile(r"\[(\w+)\]")
#     text = re.sub(pattern, sep_token, text)

#     # 3. otherwise assume the question is a sentence
#     return text

def get_output(df_input, counterfactual_method, args):
    df_input = df_input.copy()
    output_data = {
        "original_text": [],
        "original_score": [],
        "original_perplexity": [],
        "counterfactual_text": [],
        "counterfactual_score": [],
        "counterfactual_perplexity": [],
        "found_flip": [],
        "levenshtein_similarity_score": []
    }
    for i in range(len(df_input)):
        if dataset == "sst_2":
            original_text = df_input.iloc[i]["original_text"]
            original_text = truncate_text(original_text)
            original_text = format_sentence(original_text, dataset)
            print(f"Processing input {i + 1}/{len(df_input)}: num tokens: {len(original_text.split())}")

            original_score = calculate_score(original_text, sentiment_model_tokenizer, dataset, device)
            original_perplexity = calculate_perplexity(original_text)

            args = {**args, "original_score": original_score}
            counterfactual_text = counterfactual_method(original_text, calculate_score, args)
            counterfactual_text = format_sentence(counterfactual_text, dataset)

            label_width = 20
            print(f"\n{'original_text:'.ljust(label_width)} {original_text}")
            print(f"{'counterfactual_text:'.ljust(label_width)} {counterfactual_text}\n")

            counterfactual_score = calculate_score(counterfactual_text, sentiment_model_tokenizer, dataset, device)
            counterfactual_perplexity = calculate_perplexity(counterfactual_text)
            found_flip = is_flip(original_score, counterfactual_score)
            levenshtein_similarity_score = get_levenshtein_similarity_score(original_text, counterfactual_text)

            output_data["original_text"].append(original_text)
            output_data["original_score"].append(original_score)
            output_data["original_perplexity"].append(original_perplexity)
            output_data["counterfactual_text"].append(counterfactual_text)
            output_data["counterfactual_score"].append(counterfactual_score)
            output_data["counterfactual_perplexity"].append(counterfactual_perplexity)
            output_data["found_flip"].append(found_flip)
            output_data["levenshtein_similarity_score"].append(levenshtein_similarity_score)

        elif dataset == "qnli":
            row = df_input.iloc[i]
            original_question, original_answer = row["original_question"], row["original_answer"]
            original_text = f"{original_question} [SEP] {original_answer}"
            original_text = format_sentence(original_text, dataset)

            print(f"Processing input {i + 1}/{len(df_input)}: num tokens: {len(original_text.split())}")

            original_score = calculate_score(original_text, sentiment_model_tokenizer, dataset, device)
            original_perplexity = calculate_perplexity(original_text)

            args = {**args, "original_score": original_score}
            counterfactual_text = counterfactual_method(original_text, calculate_score, args)
            if counterfactual_method.__name__ == "generate_polyjuice_counterfactual":
                counterfactual_text = format_polyjuice_output(counterfactual_text, original_question, original_answer)
            counterfactual_text = format_sentence(counterfactual_text, dataset)

            label_width = 20
            print(f"\n{'original_text:'.ljust(label_width)} {original_text}")
            print(f"{'counterfactual_text:'.ljust(label_width)} {counterfactual_text}\n")

            counterfactual_score = calculate_score(counterfactual_text, sentiment_model_tokenizer, dataset, device)
            counterfactual_perplexity = calculate_perplexity(counterfactual_text)
            found_flip = is_flip(original_score, counterfactual_score)
            levenshtein_similarity_score = get_levenshtein_similarity_score(original_text, counterfactual_text)

            output_data["original_text"].append(original_text)
            output_data["original_score"].append(original_score)
            output_data["original_perplexity"].append(original_perplexity)
            output_data["counterfactual_text"].append(counterfactual_text)
            output_data["counterfactual_score"].append(counterfactual_score)
            output_data["counterfactual_perplexity"].append(counterfactual_perplexity)
            output_data["found_flip"].append(found_flip)
            output_data["levenshtein_similarity_score"].append(levenshtein_similarity_score)

    df_output = pd.DataFrame(output_data)
    return df_output


# %%
all_word_embeddings = get_all_embeddings(sentiment_model, sentiment_model_tokenizer).to(device)

# %%
from openai import OpenAI
# from google.colab import userdata

# client = OpenAI(api_key=userdata.get("API_KEY"))
client = OpenAI()

# %% [markdown]
# Test the accuracy of the model.

# %%
correct = 0

for i in range(len(df_input)):
    print(f"i: {i}")
    row = df_input.iloc[i]

    if dataset == "sst_2":
        original_text, original_label = row["original_text"], row["original_label"]
    elif dataset == "qnli":
        original_question, original_answer, original_label = row["original_question"], row["original_answer"], row["original_label"]
        original_text = f"{original_question} [SEP] {original_answer}"

    score = calculate_score(original_text, sentiment_model_tokenizer, dataset, device)
    y_hat = 1 if score >= 0.5 else 0
    if y_hat == original_label:
        correct += 1

accuracy = correct / len(df_input)
print(f"accuracy: {accuracy}")

# %% [markdown]
# ## Counterfactual generator functions

# %%
# %cd "CLOSS"
# %cd ..
%pwd

# %%
from CLOSS.closs import generate_counterfactual
import re

def generate_polyjuice_counterfactual(original_text, _, args):
    ctrl_code = None if dataset == "qnli" else "negation"
    perturbations = pj.perturb(
        orig_sent=original_text,
        ctrl_code=ctrl_code,
        num_perturbations=1,
        perplex_thred=None
    )
    counterfactual_text = perturbations[0]
    return counterfactual_text

def generate_closs_counterfactual(original_text, calculate_score, args):
    # TODO: move target label from inside CLOSS to here for AG News dataset
    counterfactual_text = generate_counterfactual(
        original_text,
        sentiment_model,
        LM_model,
        calculate_score,
        sentiment_model_tokenizer,
        all_word_embeddings,
        device,
        args
    )
    return counterfactual_text

def call_openai_api(system_prompt, model):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt}
        ],
        top_p=1,
        temperature=0.4,
        frequency_penalty=1.1
    )
    output = completion.choices[0].message.content
    return output

def generate_naive_fizle_counterfactual(original_text, _, args):
    original_score, model = args["original_score"], args["model"]
    original_label = 1 if original_score >= 0.5 else 0
    cf_label = 0 if original_label == 1 else 1

    original_label_text = id2label[original_label]
    cf_label_text = id2label[cf_label]

    system_prompt = f"""In the task of {fizle_task}, a trained black-box classifier correctly predicted the label '{original_label}' ({original_label_text}) for the following text. Generate a counterfactual explanation by making minimal changes to the input text, so that the label changes from '{original_label}' ({original_label_text}) to '{cf_label}' ({cf_label_text}). Use the following definition of 'counterfactual explanation': "A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome." Enclose the generated text within <new> tags.
    -
    Text: {original_text}"""

    print("system_prompt: ", system_prompt)

    correct_output_format = False
    for i in range(10):
        print(f"attempt: {i + 1}")
        output = call_openai_api(system_prompt, model)
        counterfactual_text = re.search("<new>(.*?)</new>", output).group(1)
        if counterfactual_text:
            correct_output_format = True
            break

    if not correct_output_format:
        print("Failed to generate counterfactual surrounded by <new> tags")
        counterfactual_text = output[5:-6]

    return counterfactual_text

def generate_guided_fizle_counterfactual(original_text, _, args):
    original_score, model = args["original_score"], args["model"]
    original_label = 1 if original_score >= 0.5 else 0
    cf_label = 0 if original_label == 1 else 1
    system_prompt = ""

    # 1. Find important words
    step1_system_prompt = " ".join([
        f"In the task of {fizle_task}, a trained black-box classifier correctly predicted the label '{original_label}' for the following text.",
        f"Explain why the model predicted the '{original_label}' label by identifying the words in the input that caused the label. List ONLY the words as a comma separated list.",
        f"\n-\nText: {original_text}",
        f"\nImportant words identified: "
    ])
    system_prompt += step1_system_prompt
    important_words = call_openai_api(step1_system_prompt, model)
    system_prompt += important_words + "\n"

    # 2. Generate the final counterfactual
    correct_output_format = False
    for i in range(10):
        step2_system_prompt = " ".join([
            f"Generate a counterfactual explanation for the original text by ONLY changing a minimal set of the words you identified, so that the label changes from '{original_label}' to '{cf_label}'.",
            f"Use the following definition of 'counterfactual explanation': 'A counterfactual explanation reveals what should have been different in an instance to observe a diverse outcome.'",
            f"Enclose the generated text within <new> tags."
        ])
        final_system_prompt = system_prompt + step2_system_prompt
        print(f"final_system_prompt: {final_system_prompt}")
        step2_output = call_openai_api(final_system_prompt, model)
        counterfactual_text = re.search("<new>(.*?)</new>", step2_output).group(1)
        if counterfactual_text:
            correct_output_format = True
            break

    if not correct_output_format:
        print("Failed to generate counterfactual surrounded by <new> tags")
        counterfactual_text = step2_output[5:-6]

    return counterfactual_text


# %%
# def generate_naive_fizle_counterfactual(original_text, calculate_score, args):

original_text = "what does umc stand for? [SEP] founded in 1968 by the union of the methodist church (usa) and the evangelical united brethren church, the umc traces its roots back to the revival movement of john and charles wesley in england as well as the great awakening in the united states."
args = {"original_score": 0.95, "model": "gpt-4-turbo"}
counterfactual_text = generate_naive_fizle_counterfactual(original_text, calculate_score, args)

print(f"original_text: {original_text}")
print(f"counterfactual_text: {counterfactual_text}")

# %%
# def calculate_score(text, sentiment_model_tokenizer, dataset, device):

original_output = calculate_score(original_text, sentiment_model_tokenizer, "qnli", device)
counterfactual_output = calculate_score(counterfactual_text, sentiment_model_tokenizer, "qnli", device)

print(f"original_output: {original_output}, counterfactual_output: {counterfactual_output}")

# %%
# s = "What came into force after the new constitution was herald? [SEP] As of that day, the new constitution heralding the Second Republic came into force."
# s = "What is the minimum required if you want to teach in Canada? [SEP] Teaching in Canada requires a post-secondary degree Bachelor's Degree."
# s = "I really hated the movie."
# s = "I really loved the movie and thought it was one of the best I've ever seen."

# s = "what came into force after the new constitution was herald? [SEP] as of that day, the new constitution heralding the second republic came into force."
s = "what came into force after the new constitution was heralded? [SEP] as of that day, the new constitution heralding the second republic came into force."

args = {
    "beam_width": 15,
    "w": 5,
    "K": 30,
    "tree_depth": 0.15,
    "substitution_evaluation_method": "hotflip_only",
    "substitution_gen_method": "hotflip_only",
    "dataset": dataset
}

# args = {
#     "beam_width": 15,
#     "w": 5,
#     "K": 30,
#     "tree_depth": 0.5,
#     "substitution_evaluation_method": "SVs",
#     "substitution_gen_method": "no_opt_lmh",
#     "dataset": dataset
# }

original_score = calculate_score(s, sentiment_model_tokenizer, dataset, device)
print(f"original_score: {original_score}")

counterfactual = generate_closs_counterfactual(s, calculate_score, args)
print(f"counterfactual: {counterfactual}")

counterfactual_score = calculate_score(counterfactual, sentiment_model_tokenizer, dataset, device)
print(f"counterfactual_score: {counterfactual_score}")

# %% [markdown]
# ## Run CLOSS and HotFlip
# 
# First run the method without optimization (`CLOSS-EO`) and without retraining the language modeling head.
# 
# - `CLOSS-EO:` skip optimizing the embedding. This increases failures but lowers perplexity.
# - `CLOSS-RTL:` skip retraining the language modeling head. This has no effect on perplexity but increases the failure rate.

# %% [markdown]
# Move to the main parent directory.

# %%
# %cd "CLOSS"
# %cd ..
%pwd

# %%
df_input.head()

# %% [markdown]
# 1. Run HotFlip:

# %%
args = {
    "beam_width": 15,
    "w": 5,
    "K": 30,
    "tree_depth": 0.15,
    "substitution_evaluation_method": "hotflip_only",
    "substitution_gen_method": "hotflip_only",
    "dataset": dataset
}

df_output = get_output(df_input, generate_closs_counterfactual, args)

# %%
df_output.head()

# %%
df_output.to_csv(f"./output/hotflip-output-{dataset}.csv", index=False)

# %% [markdown]
# 2. Run CLOSS without optimization and without retraining the language modeling head:

# %%
args = {
    "beam_width": 15,
    "w": 5,
    "K": 30,
    "tree_depth": 0.15,
    "substitution_evaluation_method": "SVs",
    "substitution_gen_method": "no_opt_lmh",
    "dataset": dataset
}

df_output = get_output(df_input, generate_closs_counterfactual, args)

# %%
df_output.head()

# %%
df_output.to_csv(f"./output/closs-output-{dataset}.csv", index=False)

# %% [markdown]
# ## Run Polyjuice

# %% [markdown]
# ### Setup

# %%
%cd polyjuice
%pwd

# %%
!python -m spacy download en_core_web_sm

# %%
%pip install -e .

# %% [markdown]
# Make sure the model is being imported properly.

# %%
import importlib
import polyjuice

importlib.reload(polyjuice)
print(polyjuice.__file__)

# %%
from polyjuice import Polyjuice

pj = Polyjuice(model_path="uw-hai/polyjuice", is_cuda=True)

# %%
text = "julia is played with exasperating blandness by laura regan ."
perturbations = pj.perturb(
    orig_sent=text,
    ctrl_code="negation",
    num_perturbations=5,
    # perplex_thred=None
)
perturbations

# %% [markdown]
# Run the model and get the output.

# %%
df_input.head()

# %%
df_output = get_output(df_input, generate_polyjuice_counterfactual, {})

# %%
df_output.head(10)

# %%
%cd ..
%pwd

# %%
df_output.to_csv(f"./output/polyjuice-output-{dataset}.csv", index=False)

# %% [markdown]
# ## FIZLE
# 
# Two variants:
# * Naive: uses a single prompt.
# * Guided: Uses two prompts. The first prompt identifies important words and the second prompt generates the counterfactual.
# 
# Hyperparameters:
# 
# For all LLMs, we use top_p sampling with p = 1, temperature t = 0.4 and a repetition penalty of 1.1.
# 

# %% [markdown]
# ### 1. FIZLE naive

# %%
df_input.head()

# %%
args = {"model": "gpt-4-turbo"}
df_output = get_output(df_input, generate_naive_fizle_counterfactual, args)

# %%
df_output.head()

# %%
df_output.to_csv(f"./output/fizlenaive-output-{dataset}-new-2.csv", index=False)

# %% [markdown]
# ### FIZLE guided

# %%
df_input.head()

# %%
args = {"model": "gpt-4-turbo"}
df_output = get_output(df_input, generate_naive_fizle_counterfactual, args)

# %%
df_output.head()

# %%
df_output.to_csv(f"./output/fizleguided-output-{dataset}.csv", index=False)


