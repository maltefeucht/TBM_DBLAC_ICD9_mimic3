###############################################################
# This file contains various functions used acroos the project
###############################################################

import os
import datetime
import torch as th


def save_results(results, log_dir, log_name):
    save_path = os.path.join(log_dir+"_" + log_name + "_results.txt")
    with open(save_path, 'w') as file:
        file.write(str(results))
        file.close()



def save_model(model, model_dir, model_name):

    timestamp = datetime.datetime.now().strftime(format="%d_%m_%Y_%H:%M:%S")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"created directory {model_dir}")

    save_path = os.path.join(model_dir, model_name + "_" + timestamp + ".pth")

    print(f"saving model to {save_path}...")
    th.save(model.state_dict(), save_path)

###########################################################
# The following two methods are applied for bert-h_lrc only
###########################################################
def my_collate1(batches):
    # return batches
    return [{key: th.stack(value) for key, value in batch.items()} for batch in batches]

def get_chunks(batch):
    # catch chunks for each sample in batch
    ids = [data["ids"] for data in batch]
    mask = [data["mask"] for data in batch]
    targets = [data["targets"] for data in batch]
    lengt = [data['len'] for data in batch]

    # concatenate chunks for each sample in batch
    ids = th.cat(ids).to(dtype=th.long)
    mask = th.cat(mask).to(dtype=th.long)
    targets = th.cat(targets).to(dtype=th.long)
    lengt = th.cat(lengt)

    return ids, mask, targets, lengt