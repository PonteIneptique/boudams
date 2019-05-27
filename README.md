# Le Boucher d'Amsterdam

Boudams, or "Le boucher d'Amsterdam", is a deep-learning tool built for tokenizing Latin or Medieval French languages.

Inspirations, bits of code and source for being able to understand how Seq2Seq words or write my own Torch module come 
both from [Ben Trevett](https://github.com/bentrevett/pytorch-seq2seq) and [Enrique Manjavacas](https://github.com/emanjavacas/pie). 
 
The initial dataset is pretty small but if you want to build with your own, it's fairly simple : you need data in the 
following shape : `"samesentence<TAB>same sentence"` where the first element is the same than the second but with no
space and they are separated by tabs (`\t`, marked here as `<TAB>`).

Things needs a little more tweaks here and there again, I'd like to see how Attention will perform. This model is 
particulary built for OCR/HTR output from manuscripts where spaces are inconsistent.


The best architecture I find for medieval French was Conv to Linear using token categorization (WordBoundary vs 
WordContent).

```text
Train Loss: 0.004 | Perplexity:   1.004 |  Acc.: 0.566 |  Lev.: 0.037 |  Lev. / char: 0.001
 Val. Loss: 0.066 | Perplexity:   1.069 |  Acc.: 0.585 |  Lev.: 0.272 |  Lev. / char: 0.009
 Test Loss: 0.057 | Perplexity:   1.059 |  Acc,: 0.586 |  Lev.: 0.235 |  Lev. / char: 0.008
```