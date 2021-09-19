import sys
sys.path.append('/')
import torch as th
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import gensim
from nltk.tokenize import TreebankWordTokenizer
from torch.utils.data import Dataset



def replace(text):
    for ch in ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '>', '#', '+', '-', '.', '!',',', '$', '\'']:
        text = text.replace(ch, '')
    return text

class MimicIII_Dataloader(Dataset):
    """Mimic 3 Full Datatsetloader"""

    def __init__(self, dataset, mode, w2v_embeddings, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with texts and annotated labels
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.mode = mode
        self.tokenizer = TreebankWordTokenizer()
        self.w2v_model = gensim.models.Word2Vec.load(w2v_embeddings)
        self.word_vectors = self.w2v_model.wv
        self.mimic_3 = pd.read_csv(dataset)
        self.transform = transform



    def aggregate_w2v(self, sample):
        """
        Pass in word2v representations per sample: nXd where n= number of words per sample, d= Dimension of w2v embedding vector
        Compute min, max per word2vec representation, concatenate them and return vector embedding per sample: 1x2d
        """
        sample_max = sample.max(dim=0, keepdim=True)
        sample_min = sample.min(dim=0, keepdim=True)
        sample_conc = th.squeeze(th.cat((sample_max[0], sample_min[0]), dim=1))
        return sample_conc


    def __len__(self):
        return len(self.mimic_3)

    def __getitem__(self, idx):
        # get HADM_IDS
        hadm_id = self.mimic_3.iloc[idx, 1]
        # get texts
        text_plain = self.mimic_3.iloc[idx, 2]
        # instantiate tokenizer
        # Embed word with word2vec pretrained embeddings
        text = self.mimic_3.iloc[idx, 2]
        text = self.tokenizer.tokenize(text)
        text = th.from_numpy(self.word_vectors[text])
        text = self.aggregate_w2v(text)
        # get labels
        label = self.mimic_3.iloc[idx, 4:-1]
        label = np.asarray(label).astype('float')
        label = th.from_numpy(label)

        if self.mode == False:
            sample = {'text_embedding':text,'labels': label}
        else:
            sample = {'text_embedding':text,'labels': label, 'discharge_summary': text_plain, 'HADM_ID': hadm_id }
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

        print(("Input_ids shape:", split[i]['text_embedding'].shape), ("labels shape:", split[i]['labels'].shape))
        if i == 0:
            break





