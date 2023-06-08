from typing import Callable, List, Optional

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding)

from metrics.utils import split_into_sentences


class ELECTRAScorer:
    def __init__(
        self,
        device='cuda:0',
        max_length=1024,
        checkpoint="grenlayk/electra-large-cola",
        model=None,
        tokenizer=None,
    ):
        # Set up model
        self.device = device
        self.max_length = max_length
        if checkpoint is None:
            assert model is not None
            assert tokenizer is not None
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint)
        self.model.eval()
        self.model.to(device)

    def score(
        self,
        texts: List[str],
        batch_size: int = 8,
        sent_agg_func: Optional[Callable] = None,
        return_sent_data: bool = False
    ):
        def tokenize_fn(instance):
            return self.tokenizer(instance["text"], truncation=True)

        if sent_agg_func is not None:
            # Split texts to sentenses
            text_sentences = [split_into_sentences(text) for text in texts]
            len_maps = np.cumsum([len(x) for x in text_sentences])
            sentences = [sent for text in text_sentences for sent in text]
            tokenized_data = Dataset.from_dict({"text": sentences}).map(
                tokenize_fn, remove_columns=["text"], batched=True)

            sent_probas = torch.empty(
                len(sentences), dtype=torch.float32, device=self.device)
        else:
            tokenized_data = Dataset.from_dict({"text": texts}).map(
                tokenize_fn, remove_columns=["text"], batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataloader = DataLoader(
            tokenized_data, batch_size=batch_size,
            shuffle=False, collate_fn=data_collator
        )

        probas = torch.empty(
            len(texts), dtype=torch.float32, device=self.device)

        start = 0
        end = batch_size
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch_pred = self.model(
                    **{k: v.to(self.device) for k, v in batch.items()})
                batch_probas = 1 / (1 + (-batch_pred.logits[:, 1]).exp())
                if sent_agg_func is None:
                    probas[start:end].copy_(batch_probas)
                else:
                    sent_probas[start:end].copy_(batch_probas)
                start = end
                end += batch_size

        # Aggreagate sentences scores for each text
        if sent_agg_func is not None:
            for i, end_idx in enumerate(len_maps):
                start_idx = len_maps[i - 1] if i != 0 else 0
                probas[i].copy_(sent_agg_func(sent_probas[start_idx:end_idx]))

        if return_sent_data:
            return (probas.cpu().detach().numpy(),
                    sent_probas.cpu().detach().numpy(), sentences)
        return probas.cpu().detach().numpy()
