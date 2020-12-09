import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field


class RerankerDataset(Dataset):
    def __init__(self, examples: List[Dict]):
        self.examples = examples
        

    def __getitem__(self, idx):
        return self.examples[idx]
    
    def __len__(self):
        return len(self.examples)

# prepares lm_labels from target_ids, returns examples with keys as expected by the forward method
# this is necessacry because the trainer directly passes this dict as arguments to the model
# so make sure the keys match the parameter names of the forward method
@dataclass
class T2TDataCollator:
    def __init__(self, tokenizer, cfg):
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_encodings = self.tokenizer.batch_encode_plus([[example['query'],example['doc']] for example in batch], 
                                                      padding=True, max_length=self.cfg.max_seq_length, truncation=True, return_tensors='pt')
        input_encodings['labels'] = torch.tensor([float(example['label']) for example in batch],dtype=torch.float)
        return input_encodings


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        
    return {'f1': f1}

    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }