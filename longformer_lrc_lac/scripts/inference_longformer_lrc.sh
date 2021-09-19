cd ..
python classifier_kfold_longformer_lrc.py --notraining --noDataset_full --local --nodebug --epochs=25 --k_folds=5 --inference --nolr_finder --batch_size=4