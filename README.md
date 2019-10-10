# Le Boucher d'Amsterdam

Boudams, or "Le boucher d'Amsterdam", is a deep-learning tool built for tokenizing Latin or Medieval French languages.

## How to cite

An article has been published about this work : https://hal.archives-ouvertes.fr/hal-02154122v1

```text
@unpublished{clerice:hal-02154122,
  TITLE = {{Evaluating Deep Learning Methods for Word Segmentation of Scripta Continua Texts in Old French and Latin}},
  AUTHOR = {Cl{\'e}rice, Thibault},
  URL = {https://hal.archives-ouvertes.fr/hal-02154122},
  NOTE = {working paper or preprint},
  YEAR = {2019},
  MONTH = Jun,
  KEYWORDS = {convolutional network ; scripta continua ; tokenization ; Old French ; word segmentation},
  PDF = {https://hal.archives-ouvertes.fr/hal-02154122/file/Evaluating_Deep_Learning_Methods_for_Tokenization_of_Scripta_Continua_in_Old_French_and_Latin%284%29.pdf},
  HAL_ID = {hal-02154122},
  HAL_VERSION = {v1},
}

```

## How to

Install the usual way you install python stuff: `python setup.py install` (**Python >= 3.6**)).

The config file can be kickstarted using `boudams template config.json`, we recommend using the following settings :

- `linear-conv-no-pos` for the model, as it is not limited by the input size;
- `normalize` and `lower` to `True` depending on your dataset size.

The initial dataset is pretty small but if you want to build with your own, it's fairly simple : you need data in the 
following shape : `"samesentence<TAB>same sentence"` where the first element is the same than the second but with no
space and they are separated by tabs (`\t`, marked here as `<TAB>`).


```json
{
    "name": "model",
    "max_sentence_size": 150,
    "network": {
        "emb_enc_dim": 256,
        "enc_n_layers": 10,
        "enc_kernel_size": 3,
        "enc_dropout": 0.25
    },
    "model": "linear-conv-no-pos",
    "learner": {
        "lr_grace_periode": 2,
        "lr_patience": 2,
        "lr": 0.0001
    },
    "label_encoder": {
        "normalize": true,
        "lower": true
    },
    "datasets": {
        "test": "./test.tsv",
        "train": "./train.tsv",
        "dev": "./dev.tsv",
        "random": true
    }
}
```

The best architecture I find for medieval French was Conv to Linear without POS using the following setup:

```json
{
    "network": {
        "emb_enc_dim": 256,
        "enc_n_layers": 10,
        "enc_kernel_size": 5,
        "enc_dropout": 0.25
    },
    "model": "linear-conv-no-pos",
    "batch_size": 64,
    "learner": {
        "lr_grace_periode": 2,
        "lr_patience": 2,
        "lr": 0.00005,
        "lr_factor": 0.5
    }
}
```


## Credits

Inspirations, bits of code and source for being able to understand how Seq2Seq words or write my own Torch module come 
both from [Ben Trevett](https://github.com/bentrevett/pytorch-seq2seq) and [Enrique Manjavacas](https://github.com/emanjavacas/pie). 
 
