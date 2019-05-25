input_text = """Segneur ,%% sachies que Mil et .C. .iiijxx. et .xvii. ans apries l'incarnation ihesucrist ,%.%. Au tans Innocent ,%% l'apostole de Rome ,%,%. et Phelippon ,%% roi de france ,%%
et Richart ,%% roi d'engleterre ,%,%. ot vn saint home en france ,%,%. qui ot nom Fouques de Nuelly (%.%. Cil Nuellys siet entre Nuelly sour Marne %,%. et paris )%.%. Et il estoit priessres et tenoit la perroche de la uille .%.%. Et ichil Fouques dont ie vos di
commencha a parle de diu %,%. par france %,%.
et par les autres pais entour ;%.%.
Et sachies que nostre sires fist maintes bieles miracles pour lui .%,%. et tant que la renommee de cel saint home ala ,%% tant qu'ele vint a l'apostole de Rome Innocent ;%.%. Et l'apostoles manda en france au saint home %,%. que il preechast des crois par s'auctorite ;%.%. Et apres i enuoia .i. sien cardonnal ,%% Maistre Pieron de Capes ,%% croisie ,%,%. et manda par lui le pardon tel con ie vous dirai :%.%.
Tout chil qui se croiseroient %,%. et feroient le sieruice diu .i. an en l'ost %,%[punctelev]
seroient quite de toz lor pechies quil auoient fais ,%% dont il seroient confies .%.%.
Pour che que chius pardons fu si grans ,%,%. si s'en esmurent moult li cuer des gens ,%,%. et moult s'en croisierent
pour chou que li pardons estoit si grans .§%.§%.


EN l'autre an apries que chil preudom Fouques parla de diu ,%,%[punctelev] ot .i. tournoi en champaigne ,%%
a .i. castiel qui a non Aicri .%,%. et par la grace de diu si auint ke Thiebaus ,%% quens de champaigne
et de Brie ,%% prist la crois ,%,%. et li cuens Looys de Bloys %,%. et de chartaing .%.%.
Et che fu a l'entree des Auens .%,%. et chil cuens thiebaus estoit iouenes hom et n'auoit pas plus de .xxij. ans ,%.%. Ne li cuens Looys n'auoit pas plus de .xxvij. ans .%,%. Chil doi conte ierent neueu le roi de france %,%. et cousin germain et neueu le roi d'engleterre %.%. De l'autre part .%% auoec ces .ij. contes se croisierent doi moult haut baron de france ,%.%. Symons de Montfort %,%. et Renaus de Mommirail .%.%. Moult fu grans la renommee par les terres .%,%[punctelev] quant cil doi se croisierent .§%.§%.


EN la terre le conte de champaigne se croisa Gerniers li euesques de Troies ,%,%. et li cuens Gautiers de Braine ,%.%. Joffrois de Joinuile ,%,%.
qui estoit senescaus de la tiere ,%.%.
Robiers ses freres ,%.%. Gautiers de voignori ,%.%. Gautiers de Mombelyart ,%.%.
Eustasces d'escouflans ,%.%. Guis dou plaissie %,%. et ses freres ,%% Henris D'ardillieres ,%.%. Ogiers de saint chienon ,%.%.""".replace(
    "%", "").replace("\n", " ")

print(input_text)

from boudams.tagger import Seq2SeqTokenizer
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

tokenizer = Seq2SeqTokenizer.load("/home/thibault/dev/boudams/models/linear-conv2019-05-24--14:08:58-0.0001.tar", device="cpu")
print(" ".join(tokenizer.annotate_text(input_text.replace(" ", ""))))