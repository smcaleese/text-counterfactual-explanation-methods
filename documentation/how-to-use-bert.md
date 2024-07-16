
```python
# 1. Define the model and the tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-imdb", num_labels=2, id2label=id2label, label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

# 2. Tokenize the sentence:
sentence = "I loved the movie. It was fantastic."
ids_tensor = tokenizer.encode(sentence, return_tensors="pt")

# 3. Get the logits:
with torch.no_grad():
    logits = model(ids_tensor).logits

# 4. Get the predicted label:
predicted_label = model.config.id2label[logits.argmax().item()] # "POSITIVE"
```