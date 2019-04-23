# Le Boucher d'Amsterdam

Boudams, or "Le boucher d'Amsterdam", is a Seq2Seq RNN built for tokenizing Latin languages. Right now, it's a really 
simple architecture based on [bentrevett](https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)'s
work.

The initial dataset is pretty small but if you want to build with your own, it's fairly simple : you need data in the 
following shape : `"samesentence<TAB>same sentence"` where the first element is the same than the second but with no
space and they are separated by tabs (`\t`, marked here as `<TAB>`).

Things needs a little more tweaks here and there again, I'd like to see how Attention will perform. This model is 
particulary built for OCR/HTR output from manuscripts where spaces are inconsistent.

Example output after 100 epochs

```text
Inp vosvenitesdevantmoiqantgevosdisquevosenaillissiezousece
Exp vos venites devant moi qant ge vos dis que vos en aillissiez ou se ce
Out  UNK vos en iient eevant si mout dolez vos diques vos vous li vois que ves

Inp nonlicuersmepartiroitelventrecarjaienvostotemisel
Exp non li cuers me partiroit el ventre car j ai en vos tote mise l
Out  UNK nol li chese prpartir ot et volentre a  it i ne vos a mes e  l

Inp amorquemereporroitmetreensonanfantsinesaicommentgemen
Exp amor que mere porroit metre en son anfant si ne sai comment ge m en
Out  UNK amour de me par orit treter en sa contant s ' en Martin , mene en n

Inp puisseconsirrerdevosennulefincarmoutmegreveraaucuerMaismiauz
Exp puisse consirrer de vos en nule fin car mout me grevera au cuer Mais miauz
Out  UNK puis ce son reirore de vos nn le france au char et mautres que vaus au mis

Inp aingeassoffrirmagrantmesaisequevosperdissiezparmoisihautanorcomme
Exp ain ge assoffrir ma grant mesaise que vos perdissiez par moi si haut anor comme
Out  UNK aigneil s fort framarant mes ais que plus avoit ares autres aarme an an ee one

Inp dechevalerieetgecuitqueeleiserabienemploieeEtsevossaviez
Exp de chevalerie et ge cuit que ele i sera bien emploiee Et se vos saviez
Out  UNK de ce aaveriere ee fuit qi eelle estoient en ar pee ee  ee ves avoiess

Inp quifuvostresperesnedeqexgenzvostreslignagesestestraizdeparla
Exp qui fu vostres peres ne de qex genz vostres lignages est estraiz de par la
Out  UNK qui fu vostre peres en de nuue ne molt estes genz a granz destrier a parr

Inp merevosnavriezpaspaorsicomgecuitdestreprozdomcarnus
Exp mere vos n avriez pas paor si com ge cuit d estre prozdom car nus
Out  UNK me vers n avoir a pas pprovris d de puist ferme mout par eon auuz

Inp quidetellignagefustnedevroitpasavoircoragedemauveitiéMaisvosn
Exp qui de tel lignage fust ne devroit pas avoir corage de mauveitié Mais vos n
Out  UNK qui de tel gengage  qui tenre avoit por sa moit ee de par la voit an avesss

Inp ansavroizoresplustantquemavolentezsoitnejaplusnemenquerez
Exp an savroiz ores plus tant que ma volentez soit ne ja plus ne m enquerez
Out  UNK anz avoir sor ne plus aant qe sa volent men en  a sau peen ee  ee  eesz
``` 

Examples after 50 Epochs, 4 layers, 512 hidden :

```text
Inp vosvenitesdevantmoiqantgevosdisquevosenaillissiezousece
Exp vos venites devant moi qant ge vos dis que vos en aillissiez ou se ce
Out  UNK vos venir et devant mait noit vos de quise  a soulles aus ee  e  ee
Inp nonlicuersmepartiroitelventrecarjaienvostotemisel
Exp non li cuers me partiroit el ventre car j ai en vos tote mise l
Out  UNK nollin  de cer a rait et solier et car done  ain et se  e  l le
Inp amorquemereporroitmetreensonanfantsinesaicommentgemen
Exp amor que mere porroit metre en son anfant si ne sai comment ge m en
Out  UNK aoy que de seroit aot de re eenon nos avant a si ee commangne e ne
Inp puisseconsirrerdevosennulefincarmoutmegreveraaucuerMaismiauz
Exp puisse consirrer de vos en nule fin car mout me grevera au cuer Mais miauz
Out  UNK puis se conserre de vos en ne li dera a tort de port e se  ait a autres a
Inp aingeassoffrirmagrantmesaisequevosperdissiezparmoisihautanorcomme
Exp ain ge assoffrir ma grant mesaise que vos perdissiez par moi si haut anor comme
Out  UNK ainz espas feinz Margant messaus et des res ees ee  aroit par a sainz hom eesoi
Inp dechevalerieetgecuitqueeleiserabienemploieeEtsevossaviez
Exp de chevalerie et ge cuit que ele i sera bien emploiee Et se vos saviez
Out  UNK de chevaller prete duit que el li se paisen  a oit ee re e  a sainiez
Inp quifuvostresperesnedeqexgenzvostreslignagesestestraizdeparla
Exp qui fu vostres peres ne de qex genz vostres lignages est estraiz de par la
Out  UNK qui tusooit espereres de que ne nostre sor les grant perse et par ee roi a
Inp merevosnavriezpaspaorsicomgecuitdestreprozdomcarnus
Exp mere vos n avriez pas paor si com ge cuit d estre prozdom car nus
Out  UNK mervez ons avoir par par sa cors de que ee ee por ee  a  por  un
Inp quidetellignagefustnedevroitpasavoircoragedemauveitiéMaisvosn
Exp qui de tel lignage fust ne devroit pas avoir corage de mauveitié Mais vos n
Out  UNK qui det il eegnar et seinz de vos ar avoit compaignie , cor it si cois ins
Inp ansavroizoresplustantquemavolentezsoitnejaplusnemenquerez
Exp an savroiz ores plus tant que ma volentez soit ne ja plus ne m enquerez
Out  UNK anz avoir vos pesul aut nu  u e volent a en et li seint ee ses eensez
```