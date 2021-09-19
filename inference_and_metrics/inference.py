import pandas as pd
import torch as th



def custom_dataloader(split):
    """
    Dataloader for Inference Dataset

    Parameters
    ----------

    split: csv_file (string): Path to the csv file with texts and annotated labels

    Returns
    -------
    dataset: object (indexed datafreame ready to be passed to dataloader)
    """
    dataset = datasetloader_mimic3_longformer.MimicIII_Dataloader(split)
    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample['input_ids'].shape, sample['label'].shape)
        if i == 2:
            break
    return dataset

def print_inference_set(split):
    """
    Print dataste used for inference
    """
    dataset = split
    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample['input_ids'].shape, sample['label'].shape)
        if i == 2:
            break


def create_mlb_dict(dataframe):
    """
    Create a dictionary for the binarized labels.
    Parameters
    ----------
    dataframe Dataframe inputed to infer binarized labels

    Returns
    -------
    dict_50: Dictionary of infered labels
    """
    df = pd.read_csv(dataframe)
    df = df.drop(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'], axis=1)
    # create array of labels
    x = df.columns.values
    #print("Array of labels is:", x)

    dict_50 = {i: x[i] for i in range(0, len(x))}
    print("Dictionary of labels is:", dict_50)

    return dict_50


def predict_ICD_codes(y_pred, y_true, dict):
    """
    Infer the labels according to the dictionary-
    Parameters
    ----------
    y_pred: tensor (sample x num_labels)
    y_true: (sample x num_labels)
    dict: dictionary holding to map the binarized labels back
    """
    assert len(y_pred) == len(dict), "y_pred must be of same length as dict"
    # Print predicted labels as a list
    predicted_labels = []
    for counter, value in enumerate(y_pred):
        if value == 1:
            predicted_labels.append(dict[counter])
            #print(dict[counter])
    #print('Predicted labels:', predicted_labels)

    # Print true labels as a list
    true_labels = []
    for counter, value in enumerate(y_true):
        if value == 1:
            true_labels.append(dict[counter])
            # print(dict[counter])4
    #print('True labels:', true_labels)

    return predicted_labels, true_labels


def compute_attention_scores(attention_weights, y_pred):
    """
    This function aggregates the attention scores only for the predicted labels, i.e. it takes only the attention scores for a label, where a entry of y_pred=1, and concatenates them into a attention score matrix of shape number of predcited labely x Seauence length.

    Parameters
    ----------
    attention_weights: Vector fo shape M x Sequence length
    y_pred: vector of shape M
    Returns: attention_scor matrix of shape #predicted labels x input sequence length
    -------
    """
    # get the indexes for the predicted labels
    index = []
    for counter, value in enumerate(y_pred):
        if value == 1:
            index.append(counter)
    # choose only attention vectors for which a label is predicted and return top-k positions of the highest attention scores
    index = th.tensor(index, dtype=th.int64)
    attention_scores = th.index_select((th.topk(attention_weights, 30).indices), 1, index)

    return attention_scores










