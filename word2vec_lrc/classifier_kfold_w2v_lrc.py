# Import Statements
import sys
import csv
sys.path.append('../')
from absl import app, flags
import collections
import constants_mimic3
from constants_mimic3 import MIMIC_3_DIR, MIMIC_3_DIR_VM, PROJECT_DIR, PROJECT_DIR_VM
import datasetloader_mimic3_w2v as datasetloader_mimic3
from inference_and_metrics import metrics, inference

import torch as th
import pytorch_lightning as pl
import torchmetrics
import matplotlib.pyplot as plot
from sklearn.model_selection import KFold
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from utils import save_model, save_results
import warnings
warnings.filterwarnings('ignore')


# Define flags
FLAGS = flags.FLAGS
# Model related flags
flags.DEFINE_boolean('training', None, 'Indicate whether to train or test model')
flags.DEFINE_boolean('Dataset_full', None, 'Choose Full Dataset or Top 50 codes, default full dataset')
flags.DEFINE_boolean('local',None, '')
flags.DEFINE_boolean('debug',None, '')
flags.DEFINE_boolean('inference',None, '')
flags.DEFINE_integer('k_folds', 5, '')
flags.DEFINE_boolean('lr_finder',None, '')
flags.DEFINE_integer('epochs', 25, '')
flags.DEFINE_integer('batch_size', 4, '') # old=2
flags.DEFINE_float('lr', 1.41e-5, '') #old=1e-5
num_classes = 50 #8921

# data related flags: All codes
flags.DEFINE_string('dev_full_lm', constants_mimic3.dev_full_lm, 'Path to dev dataset lm ')
flags.DEFINE_string('dev_full_vm', constants_mimic3.dev_full_vm, 'Path to dev dataset vm')
flags.DEFINE_string('train_full_lm', constants_mimic3.train_full_lm, 'Path to train dataset lm ')
flags.DEFINE_string('train_full_vm', constants_mimic3.train_full_vm, 'Path to train dataset vm')
flags.DEFINE_string('test_full_lm', constants_mimic3.test_full_lm, 'Path to test dataset lm ')
flags.DEFINE_string('test_full_vm', constants_mimic3.test_full_vm, 'Path to test dataset vm')
# data related flags: Top 50 codes
flags.DEFINE_string('dev_50_lm', constants_mimic3.dev_50_lm, 'Path to dev top_50 dataset lm ')
flags.DEFINE_string('dev_50_vm', constants_mimic3.dev_50_vm, 'Path to dev top_50  ataset vm')
flags.DEFINE_string('train_50_lm', constants_mimic3.train_50_lm, 'Path to train top_50 dataset lm ')
flags.DEFINE_string('train_50_vm', constants_mimic3.train_50_vm, 'Path to train top_50 dataset vm')
flags.DEFINE_string('test_50_lm', constants_mimic3.test_50_lm, 'Path to test top_50 dataset lm ')
flags.DEFINE_string('test_50_vm', constants_mimic3.test_50_vm, 'Path to test top_50 dataset vm')

# word embedding related flags
flags.DEFINE_string('embedding_w2v_lm', constants_mimic3.embedding_w2v_lm, 'pre-trained w2v model on all words lm')
flags.DEFINE_string('embedding_w2v_vm', constants_mimic3.embedding_w2v_vm, 'pre-trained w2v model on all words vm')


flags.mark_flag_as_required('training')
flags.mark_flag_as_required('Dataset_full')
flags.mark_flag_as_required('local')
flags.mark_flag_as_required('debug')
flags.mark_flag_as_required('inference')
flags.mark_flag_as_required('lr_finder')


class classifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.linear_dense = th.nn.Linear(in_features=200, out_features=200)
        self.linear_out = th.nn.Linear(in_features=200, out_features=num_classes)
        self.dropout = th.nn.Dropout(p=0.1, inplace=False)
        self.loss = th.nn.BCEWithLogitsLoss()
        self.accuracy = torchmetrics.Accuracy()
        self.accuracy_subset = torchmetrics.Accuracy(subset_accuracy=True)
        self.f_1_micro = torchmetrics.classification.F1(num_classes=num_classes, threshold=0.5, average='micro')


    def forward(self, text_embedding):
        """
        Forward pass as definde in classification head LongformerForSequenceClassification
        :param text_embedding:
        :return:
        """
        linear_dense = self.linear_dense(text_embedding)
        dropout = self.dropout(linear_dense)
        logits =self.linear_out(dropout)
        return logits


    def training_step(self, batch, batch_idx):
        # compute logits
        logits = self.forward(batch['text_embedding'])
        # compute and return loss
        loss = self.loss(logits, batch['labels'])

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """
        Validation step of longformer model.
        Parameters
        ----------
        batch: tensor (batch_size)
        batch_idx: tensor (batch_idx)

        Returns
        -------
        output: OrderedDict (8 items)
            output dictionary containing different metrics for validation step
        """
        # compute logits
        logits = self.forward(batch['text_embedding'])
        # compute and return loss
        loss = self.loss(logits, batch['labels'])
        y_pred, y_true = metrics.prepare_outputs(logits, batch['labels'])
        accuracy = self.accuracy(y_pred, y_true)
        accuracy_subset = self.accuracy_subset(y_pred, y_true)
        # outputs
        output = collections.OrderedDict({
            'val_loss': loss, 'accuracy': accuracy, 'accuracy_subset': accuracy_subset})
        return output

    def validation_epoch_end(self, outputs):
        """
        Compute the the aggregated output metrics over all epochs of the validation step and log them.
        Parameters
        ----------
        outputs: Ordered dict output of validation step (containing different metrics)
        """
        loss = th.stack([x['val_loss'] for x in outputs]).mean()
        accuracy = th.stack([x['accuracy'] for x in outputs]).mean()
        accuracy_subset = th.stack([x['accuracy_subset'] for x in outputs]).mean()
        self.log("val_loss", loss), self.log("val_accuracy", accuracy), self.log("val_accuracy_subset", accuracy_subset)

    def test_step(self, batch, batch_idx):
        """
        Test step of longformer model.
        Parameters
        ----------
        batch: tensor (batch_size)
        batch_idx: tensor (batch_idx)

        Returns
        -------
        output: OrderedDict (8 items)
            output dictionary containing different metrics for test step
        """
        # compute logits
        logits = self.forward(batch['text_embedding'])
        # compute and return loss
        loss = self.loss(logits, batch['labels'])
        y_pred, y_true = metrics.prepare_outputs(logits, batch['labels'])

        outputs = {'y_pred': y_pred, 'y_true': y_true, 'logits': logits}
        return outputs

    def test_epoch_end(self, outputs):
        """
        Compute the the aggregated output metrics over all epochs of the test step and log them.
        Parameters
        ----------
        outputs: Ordered dict output of test step (containing different metrics)
        """
        logits = th.cat([x['logits']for x in outputs], dim=0)
        y_pred = th.cat([x['y_pred']for x in outputs], dim=0)
        y_true = th.cat([x['y_true']for x in outputs], dim=0)
        # calculate all metrics
        results = metrics.all_metrics(y_pred, y_true, logits, num_classes, top_k=5)
        self.log("Results", results)
        return results

    def test_dataloader(self):
        """
        Loads the test dataset for the test step.
        Returns
        -------
        datatset: returns the full_test_dataset or the top_50_test_dataset
        """
        if FLAGS.Dataset_full == True:
            test_full_ds = datasetloader_mimic3.MimicIII_Dataloader(FLAGS.test_full_lm if FLAGS.local else FLAGS.test_full_vm, mode=True if FLAGS.inference else False, w2v_embeddings=constants_mimic3.embedding_w2v_lm if FLAGS.local else constants_mimic3.embedding_w2v_lm)
        else:
            test_50_ds = datasetloader_mimic3.MimicIII_Dataloader(FLAGS.test_50_lm if FLAGS.local else FLAGS.test_50_vm, mode=True if FLAGS.inference else False, w2v_embeddings=constants_mimic3.embedding_w2v_lm if FLAGS.local else constants_mimic3.embedding_w2v_lm)
        test_ds = test_full_ds if FLAGS.Dataset_full else test_50_ds

        # print shape of test_ds
        datasetloader_mimic3.print_dataset(test_ds)

        return th.utils.data.DataLoader(test_ds, batch_size=FLAGS.batch_size, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)

    def configure_optimizers(self):
        """
        Configuring the optmimizer for the model.
        """
        optimizer = th.optim.Adam(
            self.parameters(),
            lr=FLAGS.lr)

        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=1,
            verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,  # Changed scheduler to lr_scheduler
            'monitor': 'val_loss'
        }


def main(_):

    # retrieve train_full_kfold_ds or train_50_kfold_ds for training
    if FLAGS.inference == False:
        if FLAGS.Dataset_full == True:
            dev_full_ds = datasetloader_mimic3.MimicIII_Dataloader(FLAGS.dev_full_lm if FLAGS.local else FLAGS.dev_full_vm, mode=True if FLAGS.inference else False)
            train_full_ds = datasetloader_mimic3.MimicIII_Dataloader(FLAGS.train_full_lm if FLAGS.local else FLAGS.train_full_vm, mode=True if FLAGS.inference else False)
            datasets_full = [dev_full_ds, train_full_ds]
            train_full_kfold_ds = th.utils.data.ConcatDataset(datasets_full)

        else:
            dev_50_ds = datasetloader_mimic3.MimicIII_Dataloader(FLAGS.dev_50_lm if FLAGS.local else FLAGS.dev_50_vm, mode=True if FLAGS.inference else False, w2v_embeddings=constants_mimic3.embedding_w2v_lm if FLAGS.local else constants_mimic3.embedding_w2v_lm)
            train_50_ds = datasetloader_mimic3.MimicIII_Dataloader(FLAGS.train_50_lm if FLAGS.local else FLAGS.train_50_vm, mode=True if FLAGS.inference else False, w2v_embeddings=constants_mimic3.embedding_w2v_lm if FLAGS.local else constants_mimic3.embedding_w2v_lm)
            datasets_50 = [dev_50_ds, train_50_ds]
            train_50_kfold_ds = th.utils.data.ConcatDataset(datasets_50)


        # k fold cross validation
        if FLAGS.training == True:
            # instantiate results list to store test results of every split of k-fold cv
            results_kfold =[]
            # perform k-fold cv
            kfold = KFold(n_splits=FLAGS.k_folds)
            for fold, (train_idx, val_idx) in enumerate(kfold.split(train_full_kfold_ds if FLAGS.Dataset_full else train_50_kfold_ds)):

                # Instantiate logger for every split of k-fold cv
                if FLAGS.local == True:
                    if FLAGS.Dataset_full == False:
                        logger = TensorBoardLogger("tb_logs", name="classifier_kfold_50_w2v_lrc_lm")
                        logger.log_hyperparams(
                            {'learning_rate': FLAGS.lr, 'epochs': FLAGS.epochs, 'batch_size': FLAGS.batch_size,
                             'num_labels': num_classes})
                    else:
                        logger = TensorBoardLogger("tb_logs", name="classifier_kfold_full_w2v_lrc_lm")
                        logger.log_hyperparams(
                            {'learning_rate': FLAGS.lr, 'epochs': FLAGS.epochs, 'batch_size': FLAGS.batch_size,
                             'num_labels': num_classes})
                else:
                    if FLAGS.Dataset_full == False:
                        logger = TensorBoardLogger("tb_logs", name="classifier_kfold_50_w2v_lrc_vm")
                        logger.log_hyperparams(
                            {'learning_rate': FLAGS.lr, 'epochs': FLAGS.epochs, 'batch_size': FLAGS.batch_size,
                             'num_labels': num_classes})
                    else:
                        logger = TensorBoardLogger("tb_logs", name="classifier_kfold_50_w2v_lrc_vm")
                        logger.log_hyperparams(
                            {'learning_rate': FLAGS.lr, 'epochs': FLAGS.epochs, 'batch_size': FLAGS.batch_size,
                             'num_labels': num_classes})

                # Instantiate model for every split of k-fold cv
                model = classifier()

                # Instantiate trainer for every split of k-fold cv
                trainer = pl.Trainer(
                    logger=logger,
                    gpus=(-1 if th.cuda.is_available() else 0),
                    max_epochs=FLAGS.epochs,
                    fast_dev_run=FLAGS.debug,
                    callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.002, patience=3, mode='min')],
                    checkpoint_callback=False,
                    accumulate_grad_batches=16,  # old config=32
                    amp_level='O2')

                # print current split
                print(f"training {fold} of {FLAGS.k_folds} folds ...")

                # split covidx_train further into train and val data
                train_ds = datasetloader_mimic3.TransformableSubset(train_full_kfold_ds if FLAGS.Dataset_full else train_50_kfold_ds, train_idx)
                val_ds = datasetloader_mimic3.TransformableSubset(train_full_kfold_ds if FLAGS.Dataset_full else train_50_kfold_ds, val_idx)

                # print shape of train_ds, val_ds
                datasetloader_mimic3.print_dataset(train_ds), datasetloader_mimic3.print_dataset(val_ds)

                # instantiate train, val dataloaders
                train_dl = th.utils.data.DataLoader(train_ds,batch_size=FLAGS.batch_size, drop_last=True, shuffle=True, num_workers=4, pin_memory=True)
                val_dl = th.utils.data.DataLoader(val_ds, batch_size=FLAGS.batch_size, drop_last=False, shuffle=False, num_workers=4, pin_memory=True)

                # Run learning rate finder
                if FLAGS.lr_finder == True:
                    lr_finder = trainer.tuner.lr_find(model, train_dataloader=train_dl,  val_dataloaders=val_dl)
                    # Plot with
                    fig = lr_finder.plot(suggest=True)
                    suggested_lr = lr_finder.suggestion()
                    print('The suggested learning rate is:', suggested_lr)
                    if FLAGS.local:
                        plot.savefig(logger.log_dir, format='png')
                    else:
                        plot.savefig(logger.log_dir, format='png')
                    break

                # Perform pl training when not in inference mode
                else:
                    trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)
                    # iteratively increase max epochs of trainer
                    trainer.max_epochs += FLAGS.epochs
                    trainer.current_epoch = trainer.current_epoch + 1
                    # test current split and retrieve test_results for split of k-fold cv
                    results = trainer.test(model)
                    results_kfold.append(results)

                    # save model of current split of k-fold cv
                    save_model(model, logger.log_dir, logger.name)

            all_metrics = metrics.compute_kfold_metrics(results_kfold, FLAGS.k_folds)
            print(all_metrics)
            save_results(all_metrics, logger.log_dir, logger.name)

        else:
            # Perform pl testing when not in inference mode
            model = classifier()
            model.load_state_dict(th.load('%s/model_files/classifier_kfold_w2v_lrc_vm_01_08_2021_17:14:27.pth' % PROJECT_DIR if FLAGS.local else PROJECT_DIR_VM, map_location=th.device('cpu')))
            print("Loading model for testing from model state dict")
            trainer = pl.Trainer(
                logger=False,
                gpus=(1 if th.cuda.is_available() else 0),
                max_epochs=FLAGS.epochs,
                fast_dev_run=FLAGS.debug,
                callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.002, patience=3, mode='min')],
                checkpoint_callback=False,
                accumulate_grad_batches=16,
                amp_level='O2')
            results = trainer.test(model)
            print(results)


    # Perform inference when in inference mode
    if FLAGS.inference == True:
        # instantiate model
        model = classifier()
        model.load_state_dict(th.load('%s/model_files/classifier_kfold_w2v_lrc_vm_01_08_2021_17:14:27.pth' % PROJECT_DIR if FLAGS.local else PROJECT_DIR_VM, map_location=th.device('cpu')))
        model.eval()

        if FLAGS.Dataset_full == True:
            inference_dataset_full = datasetloader_mimic3.MimicIII_Dataloader(constants_mimic3.test_full_lm, mode=True if FLAGS.inference else False, w2v_embeddings=constants_mimic3.embedding_w2v_lm if FLAGS.local else constants_mimic3.embedding_w2v_lm)
            dict_full = inference.create_mlb_dict('%s/notes_labeled_binarized.csv' % MIMIC_3_DIR if FLAGS.local else MIMIC_3_DIR_VM)
        else:
            inference_dataset_50 = datasetloader_mimic3.MimicIII_Dataloader(constants_mimic3.test_50_lm, mode=True if FLAGS.inference else False, w2v_embeddings=constants_mimic3.embedding_w2v_lm if FLAGS.local else constants_mimic3.embedding_w2v_lm)
            dict_50 = inference.create_mlb_dict('%s/notes_labeled_50_binarized.csv' % MIMIC_3_DIR if FLAGS.local else MIMIC_3_DIR_VM)
        # print samples loaded for inference
        #inference.print_inference_set(inference_dataset_full if FLAGS.Dataset_full else inference_dataset_50)

        # instantiate imference dataloader -> batch_size setting to 1 important and not to be changed.
        inference_dataloader = th.utils.data.DataLoader(
            inference_dataset_full if FLAGS.Dataset_full else inference_dataset_50,
            batch_size = 1,
            drop_last = False,
            shuffle = False)

        # instantiate csv writer
        if FLAGS.Dataset_full:
            with open('%s/results/word2vec_lrc/predictions_w2v_lrc_full.csv' % PROJECT_DIR if FLAGS.local else '%s/results/word2vec_lrc/predictions_w2v_lrc_full.csv' % PROJECT_DIR_VM,'w') as outfile:
                columns = ['HADM_ID', 'TEXT', 'GROUND_TRUTH', 'PREDICTIONS']
                w = csv.DictWriter(outfile, fieldnames=columns)
                w.writeheader()
        else:
            with open(
                    '%s/results/word2vec_lrc/predictions_w2v_lrc_50.csv' % PROJECT_DIR if FLAGS.local else '%s/results/word2vec_lrc/predictions_w2v_lrc_50.csv' % PROJECT_DIR_VM,
                    'w') as outfile:
                columns = ['HADM_ID', 'TEXT', 'GROUND_TRUTH', 'PREDICTIONS']
                w = csv.DictWriter(outfile, fieldnames=columns)
                w.writeheader()

                for batch in inference_dataloader:
                    logits = model.forward(batch['text_embedding'])
                    y_pred, y_true = metrics.prepare_outputs_inference(logits, batch['labels'])
                    y_pred, y_true = y_pred.squeeze().detach().numpy(), y_true.squeeze().detach().numpy()
                    y_pred, y_true  = list(y_pred), list(y_true)
                    if FLAGS.Dataset_full == True:
                        predicted_labels, true_labels = inference.predict_ICD_codes(y_pred, y_true, dict_full)
                        print("The HADM_ID is", batch['HADM_ID'])
                        print("The Input text is:", batch['discharge_summary'])
                        print('Predicted labels:', predicted_labels)
                        print('True labels:', true_labels)
                    else:
                        predicted_labels, true_labels = inference.predict_ICD_codes(y_pred, y_true, dict_50)
                        print("The HADM_ID is", batch['HADM_ID'])
                        print("The Input text is:", batch['discharge_summary'])
                        print('Predicted labels:', predicted_labels)
                        print('True labels:', true_labels)
                    w.writerow({'HADM_ID': batch['HADM_ID'], 'TEXT': batch['discharge_summary'], 'GROUND_TRUTH': true_labels,'PREDICTIONS': predicted_labels})
            outfile.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    app.run(main)





#import IPython; IPython.embed();exit(1)


