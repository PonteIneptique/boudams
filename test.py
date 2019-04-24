from boudams.tagger import Seq2SeqTokenizer

vocabulary, train_dataset, dev_dataset, test_dataset = Seq2SeqTokenizer.get_dataset_and_vocabularies(
    "data/train/train.tab", "data/dev/dev.tab", "data/test/test.tab"
)


for settings, system, batch_size in [
    (dict(hidden_size=256, emb_enc_dim=256, emb_dec_dim=256), "bi-gru", 64),
    #(dict(hidden_size=256, emb_enc_dim=256, emb_dec_dim=256, n_layers=2), "lstm", 256),
    (dict(hidden_size=128, emb_enc_dim=128, emb_dec_dim=128), "gru", 256)
]:
    tagger = Seq2SeqTokenizer.load("models/"+system+".tar")
    tagger.test(test_dataset, batch_size=batch_size)
