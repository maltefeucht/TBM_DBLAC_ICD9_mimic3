import sys
sys.path.append('/')
from pytorch_lightning.loggers import TensorBoardLogger

import torch as th
import pandas as pd
import numpy as np
import transformers
from torch.utils.data import Dataset



class MimicIII_Dataloader(Dataset):
    """Mimic 3 Full Datatsetloader"""

    def __init__(self, dataset, mode, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with texts and annotated labels
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # dataset
        self.mode = mode
        self.mimic_3 = pd.read_csv(dataset)
        self.transform = transform
        self.seq_length = 510

    def __len__(self):
        return len(self.mimic_3)

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()

        # instantiate tokenizer
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        # get input ids
        encode = tokenizer.encode_plus(self.mimic_3.iloc[idx, 2], max_length=self.seq_length, padding= 'max_length', truncation=True, return_tensors='pt')
        text = th.squeeze(encode['input_ids'])
        attention_mask = th.squeeze(encode['attention_mask'])
        # get labels
        label = self.mimic_3.iloc[idx, 4:-1]
        label = np.asarray(label)
        label = label.astype('float')

        if self.mode == False:
            sample = {'input_ids': text, 'attention_mask': attention_mask, 'label': th.from_numpy(label)}
        else:
            # get HADM_IDS
            hadm_id = self.mimic_3.iloc[idx, 1]
            # get texts
            text_plain = self.mimic_3.iloc[idx, 2]
            # get tokenized texts
            tokenized_texts = tokenizer.tokenize(self.mimic_3.iloc[idx, 2])
            sample = {'input_ids': text, 'attention_mask': attention_mask, 'label': th.from_numpy(label), 'discharge_summary': text_plain, 'HADM_ID': hadm_id, 'tokenized_discharge_summary': tokenized_texts}
        if self.transform:
            sample = self.transform(sample)
        return sample



class TransformableSubset(Dataset):
    """
    Subset of a dataset at specified indices with transform.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        transform (transform): transform on Subset
    """

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.dataset[self.indices[idx]]
        if self.transform:
            sample = self.transform(sample)

        return sample


def print_dataset(split):
    for i in range(len(split)):
        sample = split[i]
        print(i, sample['input_ids'].shape, sample['attention_mask'].shape, sample['label'].shape)
        if i == 2:
            break

