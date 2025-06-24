import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Union
from collections import OrderedDict
import pandas as pd

class HTTPRequestMultiInputDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 text_feature_names: List[str],
                 categorical_feature_names: List[str],
                 numerical_feature_names: List[str],
                 label_column: str,
                 text_specs: Dict[str, Tuple[int, int]],
                 categorical_specs: Dict[str, Tuple[int, int]],
                 max_text_len: Dict[str, int],
                 char_level_tokenization: bool = True
                 ):
        self.labels = torch.tensor(data[label_column].values, dtype=torch.long)
        self.text_features_processed = OrderedDict()
        self.cat_features_processed = OrderedDict()
        self.num_features_processed = OrderedDict()

        self.text_tokenizers = {}
        for name in text_feature_names:
            vocab_size, _ = text_specs[name]
            tokenizer = Tokenizer(num_words=vocab_size, char_level=char_level_tokenization, oov_token="<unk>")
            tokenizer.fit_on_texts(data[name])
            self.text_tokenizers[name] = tokenizer

        for name in text_specs.keys():
            if name in text_feature_names:
                sequences = self.text_tokenizers[name].texts_to_sequences(data[name])
                padded_sequences = pad_sequences(
                    sequences,
                    maxlen=max_text_len[name],
                    padding='post',
                    truncating='post'
                )
                self.text_features_processed[name] = torch.tensor(padded_sequences, dtype=torch.long)
            else:
                print(f"Warning: Text feature '{name}' specified in text_specs but not in data.")

        for name in categorical_specs.keys():
            if name in categorical_feature_names:
                self.cat_features_processed[name] = torch.tensor(data[name].values, dtype=torch.long)
            else:
                print(f"Warning: Categorical feature '{name}' specified in categorical_specs but not in data.")

        for name in numerical_feature_names:
            self.num_features_processed[name] = torch.tensor(data[name].values, dtype=torch.float32)

        num_samples = len(self.labels)
        for collection in [self.text_features_processed, self.cat_features_processed, self.num_features_processed]:
            for key, tensor in collection.items():
                if tensor.size(0) != num_samples:
                    raise ValueError(f"Mismatch in number of samples for feature '{key}': {tensor.size(0)} vs {num_samples}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text_inputs_list = [self.text_features_processed[name][idx]
                            for name in self.text_features_processed.keys()]

        categorical_inputs_list = [self.cat_features_processed[name][idx]
                                   for name in self.cat_features_processed.keys()]

        numerical_inputs_list = [self.num_features_processed[name][idx]
                                 for name in self.num_features_processed.keys()]

        label = self.labels[idx]

        return text_inputs_list, categorical_inputs_list, numerical_inputs_list, label






