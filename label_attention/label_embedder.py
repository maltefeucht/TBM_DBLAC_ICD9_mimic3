import sys
sys.path.append('/')
import torch as th
import numpy as np
import torch.nn as nn
from gensim.models import KeyedVectors
import gensim
from nltk.tokenize import TreebankWordTokenizer


class LabelEmbedder():
    def __init__(self, model_w2v, label_desc, codes_used):
        # Instantiate tokenizer and pre-trained vectors for icd codes
        self.tokenizer_codes = TreebankWordTokenizer()
        self.w2v_model_codes = gensim.models.Word2Vec.load(model_w2v)
        self.word_vectors_codes = self.w2v_model_codes.wv
        # Instantiate data
        self.codes_dict = np.load(label_desc,allow_pickle='TRUE').item()
        self.codes_used = np.load(codes_used ,allow_pickle='TRUE')
        self.codes_used = self.codes_used.tolist()
        print("The number of used labels is", len(self.codes_used))

    def clean_text(self, text):
        """
        Function to preprocess and clean text.
        """
        for ch in ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '>', '#', '+', '-', '.', '!', ',', '$', '\'']:
            text = text.replace(ch, '')
        return text

    def aggregate_w2v(self, sample):
        """
        Pass in word2v representations per label: nXd where n= number of words per label, d= Dimension of w2v embedding vector
        Compute min, max per word2vec representation, concatenate them and return vector embedding per label: 1x2d
        """
        sample_max = sample.max(dim=0, keepdim=True)
        sample_min = sample.min(dim=0, keepdim=True)
        sample_conc = th.squeeze(th.cat((sample_max[0], sample_min[0]), dim=1))
        return sample_conc

    def load_label_desc(self):
        """
        This function computes a dict containing all the used ICD-9 codes in the dataset and their respective textual descriptions in sorted order
        """
        # Set of all ICD-9 existing ICD-9 codes
        all_codes_set = set(list(self.codes_dict.keys()))
        # Set of all the ICD-9 codes used in the dataset
        used_codes_set = set(self.codes_used)
        # Compute the codes to be removed
        remove_codes_set = all_codes_set - used_codes_set
        # Delete the codes to be removed from the all ICD_9 codes to obtain only the ICD-9 codes used in the dataset
        for unwanted_codes in remove_codes_set:
            del self.codes_dict[unwanted_codes]
        # Reorder the decit of used codes according to the label order as defined by mlb.binarizer -> .npy ->before order was not sorted
        reordered_dict = {k: self.codes_dict[k] for k in self.codes_used}
        return reordered_dict


    def compute_label_emb_dict(self, codes_dict):
        """
        This methods takes the textual description of an ICD-9 codes, cleans and tokenizes it and initializes every word of the textual descriptions with the pre-trained word2vec vectors.
        If no pre-trained word vector exists for a certain word, it is initilioazed randomly
        """
        codes_dict = codes_dict
        counter_exceptions_words = 0
        counter_exceptions_labels = 0
        for k, v in codes_dict.items():
            cleaned_values = self.clean_text(v)
            tokenized_values = self.tokenizer_codes.tokenize(cleaned_values)
            # check whether a textual description exists and compute emebddings accordingly
            if len(tokenized_values) >0:
                try:
                    vectorized_values = th.from_numpy(self.word_vectors_codes[tokenized_values])
                except KeyError:
                    counter_exceptions_words += 1
                    vectorized_values = th.rand((len(tokenized_values)), vectorized_values.size(1))
            else:
                counter_exceptions_labels += 1
                vectorized_values = th.rand(5, self.word_vectors_codes.vector_size)
            # Aggregate the label embeddings using Min+Max Pooling
            aggregated_vectorized_values = self.aggregate_w2v(vectorized_values)
            codes_dict[k] = aggregated_vectorized_values
        print('For', counter_exceptions_labels,' labels, no textual descriptions exists and they are therefore initialized randomly.')
        print('For', counter_exceptions_words,' words, no pretrained word2vec embeddings were found and therefore initialized randomly.')
        return codes_dict

    def initialize_label_embedding(self):
        """
        This methods initializes the label embedding matrix of all used ICD-codes. Label word2vec representations are computed bey their textual descriptions.
        """
        # obtain dict of all used ICD-9 codes and their textual descriptions
        preprocessed_codes = self.load_label_desc()
        # computed the vector representation for every ICD-9 code using pre-trained word2vec vectors
        codes_dict = self.compute_label_emb_dict(preprocessed_codes)
        # stack the obtained label vectors into a label data matrix of dimension (M x embeddings size d_a)
        list = []
        for label in self.codes_used:
            for k, v in codes_dict.items():
                if k == label:
                    list.append(v)
        W = th.stack(list, dim=0)
        label_embedding = nn.Embedding.from_pretrained(W, freeze=False)
        return label_embedding







#test = LabelEmbedder()
#test_1 = test.load_label_desc()
#codes_dict = test.compute_label_emb_dict(test_1)
#embedding = test.initialize_label_embedding(codes_dict)
#import IPython; IPython.embed();exit(1)









'''
 def aggregate_w2v(self, sample):
        """
        Pass in word2v representations per sample: nXd where n= number of words per sample, d= Dimension of w2v embedding vector
        Compute min, max per word2vec representation, concatenate them and return vector embedding per sample: 1x2d
        """
        sample_max = sample.max(dim=0, keepdim=True)
        sample_min = sample.min(dim=0, keepdim=True)
        sample_conc = th.squeeze(th.cat((sample_max[0], sample_min[0]), dim=1))
        return sample_conc

        # Instantiate tekenizer and pre-trained vectors for icd codes
        self.tokenizer_codes = TreebankWordTokenizer()
        self.w2v_model_codes = gensim.models.Word2Vec.load(
            '/Users/maltefeucht/PycharmProjects/word2vec/mimicdata/mimic3/processed_full.w2v_labels')
        self.word_vectors_codes = self.w2v_model_codes.wv

        # get embbeded ICD codes
        codes = self.codes.iloc[idx, 0]
        codes = replace(codes)
        codes = self.tokenizer_codes.tokenize(codes)
        codes = th.from_numpy(self.word_vectors_codes[codes])
        codes = self.aggregate_w2v(codes)
        
'''