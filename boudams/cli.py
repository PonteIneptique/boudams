import click
import re
import tqdm
import json
import datetime

from boudams.tagger import BoudamsTagger
from boudams.trainer import Trainer
from boudams.encoder import LabelEncoder, DatasetIterator

from boudams.dataset import conllu, base as dataset_base, plaintext


@click.group()
def cli():
    """ Boudams is a tokenizer built on deep learning. """


@cli.group("dataset")
def dataset():
    """ Dataset related functions """


@dataset.command("convert")
@click.argument("method", type=click.Choice(['tsv', 'tsv-header', 'plain-text']))
@click.argument("output_path", type=click.Path(file_okay=False))
@click.argument("input_path", nargs=-1, type=click.Path(file_okay=True, dir_okay=False))
@click.option("--min_words", type=int, default=2, help="Minimum of words to build a line")
@click.option("--max_words", type=int, default=10, help="Maximum number of words to build a line")
@click.option("--min_char_length", type=int, default=7, help="Minimum amount of characters to build a line")
@click.option("--max_char_length", type=int, default=100, help="Maximum amount of characters to build a line")
@click.option("--random_keep", type=float, default=0.3, help="Probability to keep some words for the next sequence")
@click.option("--max_kept", type=int, default=1, help="Maximum amount of words to be kept over next sequence")
@click.option("--noise_char", type=str, default=".", help="Character to add between words for noise purposes")
@click.option("--noise_char_random", type=float, default=0.2, help="Probability to add [NOISE_CHAR] in between words")
@click.option("--max_noise_char", type=int, default=2, help="Maximum amount of [NOISE_CHAR] to add sequentially")
def convert(method, output_path, input_path, min_words, max_words, min_char_length,
             max_char_length, random_keep, max_kept, noise_char, noise_char_random, max_noise_char):
    """ Build sequence training data using files with [METHOD] format in [INPUT_PATH] and saving the
    converted format into [OUTPUT_PATH]

    If you are using `tsv-header` as a method, columns containing tokens should be named "tokens" or "form"
    """
    if method.startswith("tsv"):
        conllu.convert(
            input_path, output_path, min_words=min_words, max_words=max_words,
            min_char_length=min_char_length, max_char_length=max_char_length,
            random_keep=random_keep, max_kept=max_kept, noise_char=noise_char,
            noise_char_random=noise_char_random, max_noise_char=max_noise_char,
            dict_reader=method.endswith("header")
        )
    else:
        plaintext.convert(
            input_path, output_path, min_words=min_words, max_words=max_words,
            min_char_length=min_char_length, max_char_length=max_char_length,
            random_keep=random_keep, max_kept=max_kept, noise_char=noise_char,
            noise_char_random=noise_char_random, max_noise_char=max_noise_char
        )


@dataset.command("statistics")
@click.argument("png_path", type=click.Path(file_okay=True, dir_okay=False))
@click.argument("char_count", type=click.Path(file_okay=True, dir_okay=False))
@click.argument("input_path", nargs=-1, type=click.File(mode="r"))
def sizes(png_path, char_count, input_path):
    """ Build sequence training data using files with [METHOD] format in [INPUT_PATH] and saving the
    converted format into [OUTPUT_PATH]

    If you are using `tsv-header` as a method, columns containing tokens should be named "tokens" or "form"
    """
    import matplotlib.pyplot as plt
    from collections import Counter
    lengths = []
    counter = {
        file.name: Counter()
        for file in input_path
    }
    print(counter)
    for file in input_path:
        for l in file.readlines():
            x, truth = l.split("\t")
            words = truth.strip().split()
            lengths += [len(t) for t in words]
            counter[file.name].update(Counter(list(x)))

    fig1, ax1 = plt.subplots()
    ax1.set_title('Distribution of word sizes in the dataset')
    ax1.boxplot(lengths)
    plt.savefig(png_path)

    words = list(set(keys for cnter in counter.values() for keys in cnter))
    import csv
    with open(char_count, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(["char"] + [c for c in counter]))
        writer.writeheader()
        for word in sorted(words):
            writer.writerow(dict(char=word, **{
                c: counter[c].get(word, 0)
                for c in counter
            }))



@dataset.command("generate")
@click.argument("output_path", type=click.Path(file_okay=False))
@click.argument("input_path", nargs=-1, type=click.Path(file_okay=True, dir_okay=False))
@click.option("--max_char_length", type=int, default=100, help="Maximum amount of characters to build a line")
@click.option("--train", "train_ratio", type=float, default=0.8, help="Train ratio")
@click.option("--test", "test_ratio", type=float, default=0.1, help="Test ratio")
def generate(output_path, input_path, max_char_length, train_ratio, test_ratio):
    """ Build sequence training data using files with [METHOD] format in [INPUT_PATH] and saving the
    converted format into [OUTPUT_PATH]

    If you are using `tsv-header` as a method, columns containing tokens should be named "tokens" or "form"
    """
    dev_ratio = 1.0 - train_ratio - test_ratio
    if dev_ratio <= 0.0:
        print("Train + Test cannot be greater or equal to 1.0")
        return
    dataset_base.split(input_path, output_path, max_char_length=max_char_length,
                       ratio=(train_ratio, dev_ratio, test_ratio))
    dataset_base.check(output_path, max_length=max_char_length)

@cli.command("template")
@click.argument("filename", type=click.File(mode="w"))
def template(filename):
    """ Generates a template training model in file [FILENAME]"""
    template = {
        "name": "model",
        "max_sentence_size": 150,
        "network": {  # Configuration of the encoder
          'emb_enc_dim': 256,
          'enc_n_layers': 10,
          'enc_kernel_size': 3,
          'enc_dropout': 0.25
        },
        "model": 'linear-conv',
        "learner": {
           'lr_grace_periode': 2,
           'lr_patience': 2,
           'lr': 0.0001
        },
        "label_encoder": {
            "normalize": True,
            "lower": True
        },
        "datasets": {
            "test": "./test.tsv",
            "train": "./train.tsv",
            "dev": "./dev.tsv",
            "random": True
        }
    }
    json.dump(template, filename, indent=4, separators=(',', ': '))


@cli.command("train")
@click.argument("config_files", nargs=-1, type=click.File("r"))
@click.option("--epochs", type=int, default=100, help="Number of epochs to run")
@click.option("--batch_size", type=int, default=32, help="Size of batches")
@click.option("--device", default="cpu", help="Device to use for the network (cuda, cpu, etc.)")
@click.option("--debug", default=False, is_flag=True)
def train(config_files, epochs, batch_size, device, debug):
    """ Train one or more models according to [CONFIG_FILES] JSON configurations"""
    if debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    for config_file in config_files:
        config = json.load(config_file)

        train_path, dev_path, test_path = config["datasets"]["train"],\
                                          config["datasets"]["dev"],\
                                          config["datasets"]["test"]

        vocabulary = LabelEncoder(
            maximum_length=config.get("max_sentence_size", None),
            remove_diacriticals=config["label_encoder"].get("normalize", True),
            lower=config["label_encoder"].get("lower", True)
        )
        vocabulary.build(train_path, dev_path, test_path, debug=True)
        if debug:
            from pprint import pprint
            pprint(vocabulary.mtoi)

        # Get the datasets
        train_dataset: DatasetIterator = vocabulary.get_dataset(
            train_path, randomized=config["datasets"].get("random", True))
        dev_dataset: DatasetIterator = vocabulary.get_dataset(
            dev_path, randomized=config["datasets"].get("random", True))
        test_dataset: DatasetIterator = vocabulary.get_dataset(
            test_path, randomized=config["datasets"].get("random", True))
        print("Training %s " % config_file.name)
        print("-- Dataset informations --")
        print("Number of training examples: {}".format(len(train_dataset)))
        print("Number of dev examples: {}".format(len(dev_dataset)))
        print("Number of testing examples: {}".format(len(test_dataset)))
        print("--------------------------")

        tagger = BoudamsTagger(
            vocabulary,
            device=device, system=config["model"], out_max_sentence_length=config.get("max_sentence_size", None),
            **config["network"])
        trainer = Trainer(tagger, device=device)
        print(tagger.model)
        print()

        trainer.run(
            train_dataset, dev_dataset, n_epochs=epochs,
            fpath=config["name"] + str(datetime.datetime.today()).replace(" ", "--").split(".")[0] + ".tar",
            batch_size=batch_size, **config["learner"]
        )

        trainer.test(test_dataset, batch_size=batch_size)


@cli.command("test")
@click.argument("test_path", type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.argument("model_tar", nargs=-1, type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.option("--csv_file", default=None, type=click.File(mode="w"), help="CSV target")
@click.option("--batch_size", type=int, default=32, help="Size of batches")
@click.option("--device", default="cpu", help="Device to use for the network (cuda, cpu, etc.)")
@click.option("--debug", default=False, is_flag=True)
@click.option("--verbose", default=False, is_flag=True, help="Print classification report")
def test(test_path, model_tar, csv_file, batch_size, device, debug, verbose):
    """ Train one or more models according to [CONFIG_FILES] JSON configurations"""
    if debug:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    results = []
    for config_file in model_tar:
        model = BoudamsTagger.load(config_file, device=device)

        # Get the datasets
        test_dataset: DatasetIterator = model.vocabulary.get_dataset(test_path)

        print("Testing %s " % config_file)
        print("-- Dataset informations --")
        print("Number of testing examples: {}".format(len(test_dataset)))
        print("--------------------------")

        trainer = Trainer(model, device=device)

        scorer = trainer.test(test_dataset, batch_size=batch_size, class_report=verbose)
        print("Saving confusion matrix...")
        scorer.plot_confusion_matrix(config_file+".png")
        print(scorer.scores)
        print(scorer.report)
        r = scorer.scores._asdict()
        r["model"] = model.system
        r["file"] = config_file
        results.append(r)

    if csv_file is not None:
        import csv
        writer = csv.DictWriter(csv_file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


@cli.command("tag")
@click.argument("model", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("filename", nargs=-1, type=click.File("r"))
@click.option("--device", default="cpu", help="Device to use for the network (cuda, cpu, etc.)")
def tag(model, filename, device="cpu", batch_size=64):
    """ Tag all [FILENAME] using [MODEL]"""
    print("Loading the model.")
    model = BoudamsTagger.load(model, device=device)
    print("Model loaded.")
    remove_line = True
    spaces = re.compile("\s+")
    apos = re.compile("['’]")
    for file in tqdm.tqdm(filename):
        out_name = file.name.replace(".txt", ".tokenized.txt")
        content = file.read()  # Could definitely be done a better way...
        if remove_line:
            content = spaces.sub("", content)

        # Now, extract apostrophes, remove them, and reinject them
        apos_positions = [ i for i in range(len(content)) if content[i] in ["'", "’"] ]
        content = apos.sub("", content)

        with open(out_name, "w") as out_io:
            out = ''
            for tokenized_string in model.annotate_text(content, batch_size=batch_size):
                out = out + tokenized_string+" "

            # Reinject apostrophes
            #out = 'Sainz Tiebauz fu nez en l evesché de Troies ; ses peres ot non Ernous et sa mere, Gile et furent fra'
            true_index = 0
            for i in range(len(out) + len(apos_positions)):
                if true_index in apos_positions:
                    out = out[:i] + "'" + out[i:]
                    true_index = true_index + 1
                else:
                    if not out[i] == ' ':
                        true_index = true_index + 1

            out_io.write(out)
        # print("--- File " + file.name + " has been tokenized")


@cli.command("tag-check")
@click.argument("config_model", nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("content")
def tag_check(config_model, content, device="cpu", batch_size=64):
    """ Tag all [FILENAME] using [MODEL]"""
    for model in config_model:
        print("Loading the model.")
        tokenizer = BoudamsTagger.load(model, device=device)
        print("Model loaded.")
        print(model + "\t" +" ".join(tokenizer.annotate_text(content, batch_size=batch_size)))


@cli.command("graph")
@click.argument("model", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output", type=click.Path(file_okay=True, dir_okay=False))
@click.option("--format", default="png", type=click.Choice("png", "pdf"))
def graph(model, output, format):
    """ Draw the graph representation of a given model """
    try:
        import hiddenlayer as hl
        import hiddenlayer.pytorch_builder as torch_builder
    except ImportError:
        print("You need to install hiddenlayer (pip install hiddenlayer) for this command.")
        return
    import torch

    print("Loading the model.")
    model = BoudamsTagger.load(model, device="cpu")
    print("Model loaded.")

    tensor = torch.ones((model.out_max_sentence_length,), dtype=torch.float64)

    # Build HiddenLayer graph
    g = hl.Graph()
    hl_graph = torch_builder.import_graph(
        g,
        model.model,
        args=(
            torch.zeros([64, model.out_max_sentence_length], dtype=torch.long),
            tensor.new_full((64, ), model.out_max_sentence_length)
        ))

    # Use a different color theme
    hl_graph.theme = hl.graph.THEMES["blue"].copy()  # Two options: basic and blue
    hl_graph.save(output, format=format)


if __name__ == "__main__":
    cli()
