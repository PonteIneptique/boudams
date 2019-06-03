import mufidecode
import unidecode
"sot la gnͣt abstinence dess eintes uirges ele  ꝑla"
mufidecode.mufidecode(" sot la gnͣt abstinence dess eintes uirges ele  ꝑla")
# ' sot la gnat abstinence dess eintes uirges ele  pla'
mufidecode.mufidecode(" sot la gnͣt abstinence dess eintes uirges ele  ꝑla", join=False
# (' ',  's',  'o',  't',  ' ',  'l',  'a',  ' ',  'g',  'n',  'a',  't',  ' ',  'a',  'b',  's',  't',  'i',  'n',  'e',  'n',  'c',  'e',  ' ',  'd',  'e',  's',  's',  ' ',  'e',  'i',  'n',  't',  'e',  's',  ' ',  'u',  'i',  'r',  'g',  'e',  's',  ' ',  'e',  'l',  'e',  ' ',  ' ',  'p',  'l',  'a')
unidecode.unidecode(" sot la gnͣt abstinence dess eintes uirges ele  ꝑla")
# ' sot la gnat abstinence dess eintes uirges ele  la'