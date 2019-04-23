from boudams.tagger import Seq2SeqTokenizer
from boudams.dataset import SOS_TOKEN
import math

vocabulary, train_dataset, dev_dataset, test_dataset = Seq2SeqTokenizer.get_dataset_and_vocabularies(
    "data/train/train.tab", "data/dev/dev.tab", "data/test/test.tab"
)


tagger = Seq2SeqTokenizer(vocabulary, device="cuda")
tagger.train(train_dataset, dev_dataset, n_epochs=50)

#    src = batch.src l.198 evaluate
# AttributeError: 'BucketIterator' object has no attribute 'src'
# test_loss = tagger.test(test_dataset)
# print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

Examples = """biensavoitparsasortquemaintesfoizavoitgiteequilvandroitencora	vos venites devant moi qant ge vos dis que vos en aillissiez ou se ce
nonlicuersmepartiroitelventrecarjaienvostotemisel	non li cuers me partiroit el ventre car j ai en vos tote mise l
amorquemereporroitmetreensonanfantsinesaicommentgemen	amor que mere porroit metre en son anfant si ne sai comment ge m en
puisseconsirrerdevosennulefincarmoutmegreveraaucuerMaismiauz	puisse consirrer de vos en nule fin car mout me grevera au cuer Mais miauz
aingeassoffrirmagrantmesaisequevosperdissiezparmoisihautanorcomme	ain ge assoffrir ma grant mesaise que vos perdissiez par moi si haut anor comme
dechevalerieetgecuitqueeleiserabienemploieeEtsevossaviez	de chevalerie et ge cuit que ele i sera bien emploiee Et se vos saviez
quifuvostresperesnedeqexgenzvostreslignagesestestraizdeparla	qui fu vostres peres ne de qex genz vostres lignages est estraiz de par la
merevosnavriezpaspaorsicomgecuitdestreprozdomcarnus	mere vos n avriez pas paor si com ge cuit d estre prozdom car nus
quidetellignagefustnedevroitpasavoircoragedemauveitiéMaisvosn	qui de tel lignage fust ne devroit pas avoir corage de mauveitié Mais vos n
ansavroizoresplustantquemavolentezsoitnejaplusnemenquerez	an savroiz ores plus tant que ma volentez soit ne ja plus ne m enquerez"""

Examples = [
    tuple(line.split("\t"))
    for line in Examples.split("\n")
]

import torchtext.data


class InputDataset(torchtext.data.Dataset):
    def __init__(self, text: str, vocabulary: torchtext.data.Field):
        examples = [
            torchtext.data.Example.fromdict({"src": list(line)}, fields={"src": [("src", vocabulary)]})
            for line in text.split("\n")
        ]
        super(InputDataset, self).__init__(
            examples=examples, fields=[("src", vocabulary)]
        )

    def get_iterator(self, batch_size=256):
        return torchtext.data.Iterator(self, batch_size=batch_size, device="cuda", train=False, sort=False)


data = InputDataset("\n".join([x for x, _ in Examples]), vocabulary=vocabulary)
vocabulary.batch_first = True
treated = [
    line
    for batch_out in tagger.tag(data.get_iterator())
    for line in vocabulary.reverse(batch_out)
]

for (inp, exp), out in zip(Examples, treated):
    print("Inp " + inp)
    print("Exp " + exp)
    print("Out " + out)
