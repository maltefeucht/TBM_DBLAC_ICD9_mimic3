PAD_CHAR = "**PAD**"

###############
# local machine
###############
# TODO: -> Adjust your directories to your data directories
DATA_DIR = '/Users/maltefeucht/PycharmProjects/TBM_ICD9_mimic3/mimicdata/'
MIMIC_3_DIR = '/Users/maltefeucht/PycharmProjects/TBM_ICD9_mimic3/mimicdata/mimic3'
PROJECT_DIR = '/Users/maltefeucht/PycharmProjects/TBM_ICD9_mimic3'

# tr, dev, test datasets full labels
dev_full_lm = '%s/dev_full_binarized.csv' % MIMIC_3_DIR
train_full_lm = '%s/train_full_binarized.csv' % MIMIC_3_DIR
test_full_lm = '%s/test_full_binarized.csv' % MIMIC_3_DIR
all_codes_lm = '%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR

# tr, dev, test datasets top 50 labels
dev_50_lm = '%s/dev_50_binarized.csv' % MIMIC_3_DIR
train_50_lm = '%s/train_50_binarized.csv' % MIMIC_3_DIR
test_50_lm = '%s/test_50_binarized.csv' % MIMIC_3_DIR
top_50_codes_lm = '%s/TOP_50_CODES.csv' % MIMIC_3_DIR

# pretrained word2vec label embeddings (top 50 codes and full codes)
word2vec_model_lm = '%s/processed_full.w2v_labels' % MIMIC_3_DIR
full_codes_desc_dict_lm = '%s/description_vectors_raw.npy' % MIMIC_3_DIR
labels_top_50_lm = '%s/TOP_50_LABELS.npy' % MIMIC_3_DIR
labels_full_lm = '%s/FULL_LABELS.npy' % MIMIC_3_DIR

# pretrained word2vec words embeddings for word2vec model
embedding_w2v_lm = '%s/processed_full.w2v' % MIMIC_3_DIR

#############################
# remote machine (p3.2xlarge)
#############################
# TODO: -> Adjust your directories to your data directories
DATA_DIR_VM = '/home/ubuntu/PycharmProjects/longformer_lr/longformer_lr/mimicdata'
MIMIC_3_DIR_VM = '/home/ubuntu/PycharmProjects/longformer_lr/longformer_lr/mimicdata/mimic3'
PROJECT_DIR_VM = '/home/ubuntu/PycharmProjects/longformer_lr/longformer_lr'


# tr, dev, test datasets full labels
dev_full_vm = '%s/dev_full_binarized.csv' % MIMIC_3_DIR_VM
train_full_vm = '%s/train_full_binarized.csv' % MIMIC_3_DIR_VM
test_full_vm = '%s/test_full_binarized.csv' % MIMIC_3_DIR_VM
all_codes_vm = '%s/ALL_CODES_filtered.csv' % MIMIC_3_DIR_VM

# tr, dev, test datasets top 50 labels
dev_50_vm = '%s/dev_50_binarized.csv' % MIMIC_3_DIR_VM
train_50_vm = '%s/train_50_binarized.csv' % MIMIC_3_DIR_VM
test_50_vm = '%s/test_50_binarized.csv' % MIMIC_3_DIR_VM
top_50_codes_vm = '%s/TOP_50_CODES.csv' % MIMIC_3_DIR_VM

# pretrained word2vec label embeddings (top 50 codes and full codes)
word2vec_model_vm = '%s/processed_full.w2v_labels' % MIMIC_3_DIR_VM
full_codes_desc_dict_vm = '%s/description_vectors_raw.npy' % MIMIC_3_DIR_VM
labels_top_50_vm = '%s/TOP_50_LABELS.npy' % MIMIC_3_DIR_VM
labels_full_vm = '%s/FULL_LABELS.npy' % MIMIC_3_DIR_VM

# pretrained word2vec words embeddings for word2vec model
embedding_w2v_vm = '%s/processed_full.w2v' % MIMIC_3_DIR_VM