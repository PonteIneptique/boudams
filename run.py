from boudams.tagger import Seq2SeqTokenizer

vocabulary, train_dataset, dev_dataset, test_dataset = Seq2SeqTokenizer.get_dataset_and_vocabularies(
    #"data/fro/train.tsv", "data/fro/dev.tsv", "data/fro/test.tsv"
    "data/small/train.tsv", "data/small/dev.tsv", "data/small/test.tsv"
)
from pprint import pprint
#pprint(vocabulary.vocab.freqs)
print("-- Dataset informations --")
print("Number of training examples: {}".format(len(train_dataset.examples)))
print("Number of dev examples: {}".format(len(dev_dataset.examples)))
print("Number of testing examples: {}".format(len(test_dataset.examples)))
print("--------------------------")


def examples(obj):
    example_sentences = [
        ('vosvenitesdevantmoiqantgevosdisquevosenaillissiezousece', 'vos venites devant moi qant ge vos dis que vos en aillissiez ou se ce'),
        ('nonlicuersmepartiroitelventrecarjaienvostotemisel', 'non li cuers me partiroit el ventre car j ai en vos tote mise l'),
    ]
    treated = obj.annotate([x[0] for x in example_sentences])

    for (inp, exp), out in zip(example_sentences, treated):
        print("Inp " + inp)
        print("Exp " + exp)
        print("Out " + out)
        print("----\n\n")


for settings, system, batch_size in [
    #(dict(hidden_size=512, emb_enc_dim=256, emb_dec_dim=256), "conv", 64),
    #(dict(hidden_size=256, emb_enc_dim=256, emb_dec_dim=256, n_layers=2), "lstm", 256),
    #(dict(hidden_size=128, emb_enc_dim=128, emb_dec_dim=128), "gru", 256),
    (dict(hidden_size=512, emb_enc_dim=256, emb_dec_dim=256), "bi-gru", 64)
]:
    tagger = Seq2SeqTokenizer(vocabulary, device="cuda", system=system, **settings)
    print(tagger.model)
    tagger.train(train_dataset, dev_dataset, n_epochs=20, fpath="models/"+system+"-2.tar", batch_size=batch_size,
                 after_epoch_fn=examples)

#    src = batch.src l.198 evaluate
# AttributeError: 'BucketIterator' object has no attribute 'src'
# test_loss = tagger.test(test_dataset)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
