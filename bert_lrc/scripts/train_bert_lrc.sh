cd ..
python classifier_kfold_bert_lrc.py --training --noDataset_full --local --nodebug --epochs=25 --k_folds=5 --noinference --nolr_finder --batch_size=4