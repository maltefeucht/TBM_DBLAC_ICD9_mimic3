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

    def __init__(self, dataset, mode, max_len=8000, min_len=1, chunk_len=512, overlap_len=50, approach="all",transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with texts and annotated labels
            transform (callable, optional): Optional transform to be applied on a sample.
            TODO: max_len is not used, therefore in tokenization all overflowing tokens returned and full text lengths processed
        """
        self.mode = mode
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len
        self.overlap_len = overlap_len
        self.chunk_len = chunk_len
        self.approach = approach
        self.min_len = min_len
        self.mimic_3 = pd.read_csv(dataset)
        self.transform = transform
        assert self.chunk_len > self.overlap_len, "chunk_len must be bigger than overlap_len"

    def long_terms_tokenizer(self, data_tokenize, targets):
        """  tranfrom tokenized data into a long token that take care of
        overflow token according to the specified approch.

        Parameters
        ----------
        data_tokenize: dict
        tokenized results of a sample from bert tokenizer encode_plus method.

        targets: tensor
            labels of each samples.

        Returns
        _______
        long_token: dict
            a dictionnary that contains
             - [ids]  tokens ids
             - [mask] attention mask of each token
             - [token_types_ids] the type ids of each token. note that each token in the same sequence has the same type ids
             - [targets_list] list of all sample label after add overlap token as sample according to the approach used
             - [len] length of targets_list
        """

        long_terms_token = []
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []

        previous_input_ids = data_tokenize["input_ids"].reshape(-1)
        previous_attention_mask = data_tokenize["attention_mask"].reshape(-1)
        previous_token_type_ids = data_tokenize["token_type_ids"].reshape(-1)
        remain = data_tokenize.get("overflowing_tokens")
        #.flip(dims=[0, 1])
        remain = remain.reshape(-1)
        targets = targets

        input_ids_list.append(previous_input_ids)
        attention_mask_list.append(previous_attention_mask)
        token_type_ids_list.append(previous_token_type_ids)
        targets_list.append(targets)

        if remain.numel() !=0 and self.approach != 'head':
            remain = remain.long()
            idxs = range(len(remain)+self.chunk_len)
            idxs = idxs[(self.chunk_len-self.overlap_len-2)::(self.chunk_len-self.overlap_len-2)]
            input_ids_first_overlap = previous_input_ids[-(self.overlap_len+1):-1]
            start_token = th.tensor([101], dtype=th.long)
            end_token = th.tensor([102], dtype=th.long)

            for i, idx in enumerate(idxs):
                if i == 0:
                    input_ids = th.cat((input_ids_first_overlap, remain[:idx]))
                elif i == len(idxs):
                    input_ids = remain[idx:]
                elif previous_idx >= len(remain):
                    break
                else:
                    input_ids = remain[(previous_idx-self.overlap_len):idx]
                previous_idx = idx

                nb_token = len(input_ids)+2
                attention_mask = th.ones(self.chunk_len, dtype=th.long)
                attention_mask[nb_token:self.chunk_len] = 0
                token_type_ids = th.zeros(self.chunk_len, dtype=th.long)
                input_ids = th.cat((start_token, input_ids, end_token))
                if self.chunk_len-nb_token > 0:
                    padding = th.zeros(
                        self.chunk_len-nb_token, dtype=th.long)
                    input_ids = th.cat((input_ids, padding))

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                token_type_ids_list.append(token_type_ids)
                targets_list.append(targets)
        return({
            'ids': input_ids_list,
            'mask': attention_mask_list,
            'token_type_ids': token_type_ids_list,
            'targets': targets_list,
            'len': [th.tensor(len(targets_list), dtype=th.long)]

        })

    def __len__(self):
        return len(self.mimic_3)

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()

        text = self.mimic_3.iloc[idx, 2]
        label = self.mimic_3.iloc[idx, 4:-1]
        label = np.asarray(label).astype('float')
        label = th.from_numpy(label)
        data = self.tokenizer.encode_plus(
            text,
            max_length=self.chunk_len,
            truncation = True,
            padding= 'max_length',
            #pad_to_max_length=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True,
            return_tensors='pt')

        long_token = self.long_terms_tokenizer(data, label)
        if self.mode == False:
            long_token
        else:
            # get HADM_IDS
            hadm_id = self.mimic_3.iloc[idx, 1]
            # get texts
            text_plain = self.mimic_3.iloc[idx, 2]
            meta = {'discharge_summary': text_plain, 'HADM_ID': hadm_id}
            # TODO: append metadata for inference
            return long_token

        return long_token


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
        print(("Input_ids shape:", i, sample['ids'][0].shape), ("attention_mask shape:",sample['mask'][0].shape),("labels shape:",sample['targets'][0].shape))
        if i == 0:
            break



