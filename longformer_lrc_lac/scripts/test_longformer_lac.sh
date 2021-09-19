cd ..
python classifier_kfold_longformer_lac.py --notraining --noDataset_full --local --nodebug --epochs=25 --k_folds=5 --noinference --nolr_finder --batch_size=4 --nofreeze_longformer