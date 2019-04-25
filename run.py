from boudams.tagger import Seq2SeqTokenizer

vocabulary, train_dataset, dev_dataset, test_dataset = Seq2SeqTokenizer.get_dataset_and_vocabularies(
    "data/fro/train.tsv", "data/fro/dev.tsv", "data/fro/test.tsv"
)

print("-- Dataset informations --")
print("Number of training examples: {}".format(len(train_dataset.examples)))
print("Number of dev examples: {}".format(len(dev_dataset.examples)))
print("Number of testing examples: {}".format(len(test_dataset.examples)))
print("--------------------------")


for settings, system, batch_size in [
    #(dict(hidden_size=256, emb_enc_dim=256, emb_dec_dim=256), "bi-gru", 64),
    (dict(hidden_size=256, emb_enc_dim=256, emb_dec_dim=256, n_layers=2), "lstm", 256),
    #(dict(hidden_size=128, emb_enc_dim=128, emb_dec_dim=128), "gru", 256)
]:
    tagger = Seq2SeqTokenizer(vocabulary, device="cuda", system=system, **settings)
    tagger.train(train_dataset, dev_dataset, n_epochs=50, fpath="models/"+system+".tar", batch_size=batch_size)

#    src = batch.src l.198 evaluate
# AttributeError: 'BucketIterator' object has no attribute 'src'
# test_loss = tagger.test(test_dataset)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
