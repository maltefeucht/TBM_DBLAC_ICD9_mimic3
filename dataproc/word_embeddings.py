"""
    Pre-train embeddings using gensim w2v implementation (CBOW by default)
"""
import gensim.models.word2vec as w2v
import csv

class ProcessedIter(object):

    def __init__(self, Type, Y, filename):
        self.filename = filename
        self.type = Type


    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if self.type == "Text":
                    yield (row[3].split())
                if self.type == "Label":
                    yield (row[1].split())


def word_embeddings(Type, Y, notes_file, embedding_size, min_count, n_iter):
    if Type == "Text":
        modelname = "processed_%s.w2v" % (Y)
    if Type == "Label":
        modelname = "processed_%s.w2v_labels" % (Y)
    sentences = ProcessedIter(Type, Y, notes_file)

    model = w2v.Word2Vec(vector_size=embedding_size, min_count=min_count, workers=4)
    print("building word2vec vocab on %s..." % (notes_file))

    model.build_vocab(sentences)
    print("training...")
    model.train(sentences, total_examples=model.corpus_count, epochs=n_iter)
    out_file = '/'.join(notes_file.split('/')[:-1] + [modelname])
    print("writing embeddings to %s" % (out_file))
    model.save(out_file)
    return out_file

