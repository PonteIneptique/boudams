from boudams.tagger import Seq2SeqTokenizer

tokenizer = Seq2SeqTokenizer.load("models/bi-gru.first-run.tar")
Examples = """vosvenitesdevantmoiqantgevosdisquevosenaillissiezousece	vos venites devant moi qant ge vos dis que vos en aillissiez ou se ce
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

treated = tokenizer.annotate([x[0] for x in Examples])

for (inp, exp), out in zip(Examples, treated):
    print("Inp " + inp)
    print("Exp " + exp)
    print("Out " + out + "\n")
