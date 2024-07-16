"BERT model for finetuning."

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from models.model import Model
from config import BERT as BERT_MODEL


class BERT(Model):
    def __init__(self,
                 hparams,
                 dataset,
                 bert_model=BERT_MODEL):
        """Finetunable BERT for sequence classification model with PyTorch-Lightning.

        Args:
            hparams: hyperparameters
            dataset: dataset containing train/dev/test instances
            bert_model: name of bert model (defaults to `bert_base_uncased`)
        """
        super().__init__(hparams, dataset)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.model = BertForSequenceClassification.from_pretrained(bert_model,
                                                                   num_labels=dataset.target_size)

    def __repr__(self):
        return 'bert'

    def encode(self, X):
        """Encode sequence of instances."""
        if isinstance(X, str):
            X = [X]
        if 'pandas' in str(type(X)):
            X = X.values
        if 'numpy' in str(type(X)):
            X = X.tolist()
        return self.tokenizer.batch_encode_plus(
            X,
            pad_to_max_length=True,
            return_tensors='pt'
        )['input_ids']

    def forward_pass(self, X):
        """Single forward pass with BERT model."""
        return self.model(X)[0]

    def configure_optimizers(self):
        """Configure optimizer and scheduler.

        Based on: https://github.com/huggingface/transformers/blob/master/examples/lightning_base.py
        """
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)

        # Scheduler
        effective_batch_size = self.hparams.batch_size * max(1, self.hparams.gpus)
        total_steps = (len(self.dataset.data['train']) / effective_batch_size) * self.hparams.max_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=total_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]
