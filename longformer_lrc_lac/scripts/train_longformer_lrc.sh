cd ..
python classifier_kfold_longformer_lrc.py --training --noDataset_full --local --nodebug --epochs=25 --k_folds=5 --noinference --nolr_finder --batch_size=4