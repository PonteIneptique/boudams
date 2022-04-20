import click
import logging
import re
import json
import datetime
from typing import List

import tqdm

from torch.utils.data import DataLoader
import pytorch_lightning as pl


from boudams.tagger import BoudamsTagger, OptimizerParams
from boudams.trainer import Trainer, logger, ACCEPTABLE_MONITOR_METRICS
from boudams.encoder import LabelEncoder
from boudams.modes import SimpleSpaceMode, AdvancedSpaceMode
from boudams.dataset import BoudamsDataset
from boudams.data_generation import base as dataset_base, plaintext, splitter as all_splitters
from boudams.utils import parse_params


_POSSIBLE_MODES = list(LabelEncoder.Modes.keys())


@click.group()
def cli():
    """ Boudams is a tokenizer built on deep learning. """


@cli.group("dataset")
def dataset():
    """ Dataset related functions """


def _get_mode(mode: str, mode_kwargs: str = "") -> SimpleSpaceMode:
    if mode == "simple-space":
        return SimpleSpaceMode()
    elif mode == "advanced-space":
        return AdvancedSpaceMode()


@dataset.command("convert")
@click.argument("splitter", type=click.Choice(['words', 'sentence']))
@click.argument("input_path", nargs=-1, type=click.Path(file_okay=True, dir_okay=False))
@click.argument("output_path", type=click.Path(file_okay=False))
@click.option("--mode", type=click.Choice(_POSSIBLE_MODES),
              default="simple-space", show_default=True,
              help="Type of encoder you want to set-up")
@click.option("--splitter-regex", type=str, default=None, show_default=True,
              help="Regular expression for some splitter")
@click.option("--min-chars", type=int, default=2, show_default=True,
              help="Discard samples smaller than min-chars")
@click.option("--min_words", type=int, default=2, show_default=True,
              help="Minimum of words to build a line [Word splitter only]")
@click.option("--max_words", type=int, default=10, show_default=True,
              help="Maximum number of words to build a line [Word splitter only]")
@click.option("--mode-ratios", type=str, default="", show_default=True,
              help="Token ratios for modes at mask generation. Eg. `keep-space=.3&fake-space=.01`"
                   "will have a 30% chance of keeping a space and a 1% one to generate fake space after each char")
def convert(output_path, input_path, mode, splitter, splitter_regex, min_words, max_words, min_chars,
            mode_ratios):
    """ Build sequence training data using files with [METHOD] format in [INPUT_PATH] and saving the
    converted format into [OUTPUT_PATH]

    If you are using `tsv-header` as a method, columns containing tokens should be named "tokens" or "form"
    """
    if splitter == "words":
        splitter = all_splitters.WordSplitter(
            min_words=min_words,
            max_words=max_words,
            **({"splitter": splitter_regex} if splitter_regex else {})
        )
    else:
        splitter = all_splitters.SentenceSplitter(
            **({"splitter": splitter_regex} if splitter_regex else {})
        )
    plaintext.convert(
        input_path, output_path,
        splitter=splitter, mode=_get_mode(mode=mode),
        min_chars=min_chars,
        token_ratio=parse_params(mode_ratios)
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
    #dataset_base.check(output_path, max_length=max_char_length)


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
@click.argument("train-set", type=click.Path(file_okay=True, exists=True, dir_okay=False))
@click.argument("dev-set", type=click.Path(file_okay=True, exists=True, dir_okay=False))
@click.argument("test-set", type=click.Path(file_okay=True, exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False, exists=False))
@click.option("--architecture", type=str, help="VGSL-Like architecture.",
              default="[E256 Pl Do.3 CSs5,256,10Do.25 L256]", show_default=True)
@click.option("--mode", type=click.Choice(_POSSIBLE_MODES),
              default="simple-space", show_default=True,
              help="Type of encoder you want to set-up")
@click.option("--normalize", type=bool, is_flag=True, default=False, help="Normalize string input with unidecode"
                                                                          " or mufidecode")
@click.option("--lower", type=bool, is_flag=True, default=False, help="Lower strings")
@click.option("--epochs", type=int, default=100, help="Number of epochs to run")
@click.option("--batch_size", type=int, default=32, help="Size of batches")
@click.option("--device", default="cpu", help="Device to use for the network (cuda:0, cpu, etc.)")
@click.option("--debug", default=False, is_flag=True)
@click.option("--auto-lr", default=False, is_flag=True, help="Find the learning rate automatically")
@click.option("--workers", default=1, type=int, help="Number of workers to use to load data")
@click.option("--metric", default="f1", type=click.Choice(ACCEPTABLE_MONITOR_METRICS), help="Metric to monitor")
@click.option("--avg", default="macro", type=click.Choice(["micro", "macro"]), help="Type of avering method to use on "
                                                                                    "metrics")
@click.option("--lr", default=.0001, type=float, help="Learning rate",
              show_default=True)
@click.option("--delta", default=.001, type=float, help="Minimum change in the monitored quantity to qualify as an "
                                                        "improvement",
              show_default=True)
@click.option("--patience", default=5, type=int, help="Number of checks with no improvement after which training "
                                                      "will be stopped",
              show_default=True)
@click.option("--lr-patience", default=3, type=int, help="Number of checks with no improvement for lowering LR",
              show_default=True)
@click.option("--shuffle/--no-shuffle", type=bool, is_flag=True, default=True,
              help="Suppress the shuffling of datasets", show_default=True)
@click.option("--lr-factor", default=.5, type=float, help="Ratio for lowering LR", show_default=True)
@click.option("--seed", default=None, type=int, help="Runs deterministic training")
@click.option("--optimizer", default="Adams", type=click.Choice(["Adams", "Ranger"]), help="Optimizer to use")
@click.option("--val-interval", default=1.0, type=float, help="How often to check the validation set. Pass a float in"
                                                              " the range [0.0, 1.0] to check after a fraction of the"
                                                              " training epoch. Pass an a number > 1 to check after a"
                                                              " fixed number of training batches.")
# ToDo: Figure out the bug with Ranger
# pytorch_lightning.utilities.exceptions.MisconfigurationException: The closure hasn't been executed. HINT: did you call
# `optimizer_closure()` in your `optimizer_step` hook? It could also happen because the
# `optimizer.step(optimizer_closure)` call did not execute it internally.
def train(
        train_set: str, dev_set: str, test_set: str,
        architecture: str, output: str, mode: str,
        normalize: bool, lower: bool,
        epochs: int, batch_size: int, device: str, debug: bool, workers: int,
        auto_lr: bool,
        metric: str, avg: str,
        lr: float, delta: float, patience: int,
        lr_patience: int, lr_factor: float,
        seed: int, optimizer: str, shuffle: bool, val_interval: float):
    """ Train one or more models according to [CONFIG_FILES] JSON configurations"""
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if seed:
        pl.seed_everything(seed, workers=True)

    device = device.lower()
    if device == 'cpu':
        device = None
    elif device.startswith('cuda'):
        device = [int(device.split(':')[-1])]
    else:
        click.echo(click.style("Device is invalid. Either use `cpu` or `cuda:0`, `cuda:1`", fg="red"))
        return

    train_path, dev_path, test_path = train_set, dev_set, test_set

    vocabulary = LabelEncoder(
        mode=mode,
        remove_diacriticals=normalize,
        lower=lower
    )
    maximum_sentence_size = vocabulary.build(train_path, dev_path, test_path, debug=True)
    if debug:
        from pprint import pprint
        pprint(vocabulary.mtoi)

    # Get the datasets
    train_dataset: BoudamsDataset = vocabulary.get_dataset(train_path)
    dev_dataset: BoudamsDataset = vocabulary.get_dataset(dev_path)
    test_dataset: BoudamsDataset = vocabulary.get_dataset(test_path)

    logger.info("Architecture %s " % architecture)
    logger.info("-- Dataset informations --")
    logger.info(f"Number of training examples: {len(train_dataset)}")
    logger.info(f"Number of dev examples: {len(dev_dataset)}")
    logger.info(f"Number of testing examples: {len(test_dataset)}")
    logger.info(f"Vocabulary Size: {len(vocabulary)}")
    logger.info("--------------------------")

    tagger = BoudamsTagger(
        vocabulary,
        architecture=architecture,
        maximum_sentence_size=maximum_sentence_size,
        metric_average=avg,
        optimizer=OptimizerParams(
            optimizer,
            kwargs={"lr": lr},
            scheduler={
                "patience": lr_patience,
                "factor": lr_factor,
                "threshold": delta
            }
        )
    )
    trainer = Trainer(
        gpus=device,
        patience=patience,
        min_delta=delta,
        monitor=metric,
        max_epochs=epochs,
        gradient_clip_val=1,
        model_name=output,
        #  n_epochs=epochs,
        auto_lr_find=auto_lr,
        deterministic=True if seed else False,
        val_check_interval=int(val_interval) if val_interval > 1.1 else val_interval
    )
    train_dataloader, dev_dataloader = (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=train_dataset.train_collate_fn,
            num_workers=workers
        ),
        DataLoader(
            dev_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dev_dataset.train_collate_fn,
            num_workers=workers
        )
    )

    if auto_lr:
        trainer.tune(tagger, train_dataloader, dev_dataloader)
        return
    trainer.fit(tagger, train_dataloader, dev_dataloader)

    trainer.test(
        tagger,
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=test_dataset.train_collate_fn,
            num_workers=workers,
            shuffle=False
        )
    )


@cli.command("test")
@click.argument("test_path", type=click.Path(dir_okay=False, file_okay=True, exists=True))
@click.argument("models", nargs=-1, type=click.Path(dir_okay=False, file_okay=True, exists=True))
#@click.option("--csv_file", default=None, type=click.File(mode="w"), help="CSV target")
@click.option("--batch_size", type=int, default=32, help="Size of batches")
@click.option("--device", default="cpu", help="Device to use for the network (cuda, cpu, etc.)")
@click.option("--debug", default=False, is_flag=True)
#@click.option("--verbose", default=False, is_flag=True, help="Print classification report")
@click.option("--workers", default=1, type=int, help="Number of workers to use to load data")
@click.option("--avg", default="macro", type=click.Choice(["micro", "macro"]), help="Type of avering method to use on "
                                                                                    "metrics")
def test(test_path, models, batch_size, device, debug, workers: int, avg: str):
    """ Test one or many [MODELS] on the file at TEST_PATH """
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if device == 'cpu':
        device = None
    elif device.startswith('cuda'):
        device = [int(device.split(':')[-1])]

    results = []
    for model in models:
        model = BoudamsTagger.load(model)
        model.add_metrics("test", avg)

        # Get the datasets
        test_dataset: BoudamsDataset = model.vocabulary.get_dataset(test_path)

        logger.info("Testing %s " % model)
        logger.info("-- Dataset informations --")
        logger.info("Number of testing examples: {}".format(len(test_dataset)))
        logger.info("--------------------------")

        trainer = Trainer(
            gpus=device
        )

        logged_values = trainer.test(
            model,
            DataLoader(
                test_dataset,
                batch_size=batch_size,
                num_workers=workers,
                collate_fn=test_dataset.train_collate_fn
            )
        )
        print(model.vocabulary.format_confusion_matrix(model.confusion_matrix.tolist()))
        import tabulate
        print("\n\n")
        print(tabulate.tabulate(
            [
                (metric.replace("test_", ""), f"{score*100:.2f}")
                for metric, score in logged_values[0].items()
            ],
            headers=("metric", "score")
        ))


@cli.command("tag")
@click.argument("model", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("filename", nargs=-1, type=click.File("r"))
@click.option("--device", default="cpu", help="Device to use for the network (cuda, cpu, etc.)")
@click.option("--batch-size", default=64, help="Batch Size")
def tag(model, filename, device="cpu", batch_size=64):
    """ Tag all [FILENAME] using [MODEL]"""
    print("Loading the model.")
    model = BoudamsTagger.load(model, device=device)
    model.eval()
    print("Model loaded.")
    for file in tqdm.tqdm(filename):
        out_name = file.name.replace(".txt", ".tokenized.txt")
        content = file.read()  # Could definitely be done a better way...
        if model.vocabulary.mode.name == "simple-space":
            content = re.sub(r"\s+", "", content)
        elif model.vocabulary.mode.NormalizeSpace:
            content = re.sub(r"\s+", " ", content)
        file.close()
        with open(out_name, "w") as out_io:
            out = ''
            for tokenized_string in model.annotate_text(
                    content,
                    batch_size=batch_size,
                    device=device
            ):
                out = out + tokenized_string + "\n"
            out_io.write(out)
        print("--- File " + file.name + " has been tokenized")


@cli.command("tag-check")
@click.argument("config_model", nargs=-1, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("content")
@click.option("--device", default="cpu", help="Device to use for the network (cuda, cpu, etc.)")
@click.option("--batch-size", default=64, help="Batch Size")
def tag_check(config_model, content, device="cpu", batch_size=64):
    """ Tag all [FILENAME] using [MODEL]"""
    for model in config_model:
        click.echo(f"Loading the model {model}.")
        boudams = BoudamsTagger.load(model, device=device)
        boudams.eval()
        click.echo(f"\t[X] Model loaded")
        click.echo("\n".join(boudams.annotate_text(content, splitter=r"([\.!\?]+)", batch_size=batch_size, device=device)))


@cli.command("graph")
@click.argument("model", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output", type=click.Path(file_okay=True, dir_okay=False))
@click.option("--format", default="png", type=click.Choice(["png", "pdf"]))
def graph(model, output, format):
    """ Draw the graph representation of a given model """
    try:
        import torchviz
    except ImportError:
        print("You need to install torchviz (pip install torchviz) for this command.")
        return
    import torch

    print("Loading the model.")
    model = BoudamsTagger.load(model, device="cpu")
    model.eval()
    print("Model loaded.")

    tensor = (torch.ones((2, 1), dtype=torch.int), torch.ones(2, dtype=torch.int64))

    print(tensor)
    out = model(*tensor)
    # Build HiddenLayer graph
    hl_graph = torchviz.make_dot(out.mean(), params=dict(model.named_parameters()))
    hl_graph.format = "png"
    hl_graph.save(output)
    hl_graph.render(filename=output)


if __name__ == "__main__":
    cli()
