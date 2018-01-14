import os

from util.dataset_util import SSUField

from collections import OrderedDict, Counter

from torchtext import data, vocab
from torchtext import datasets
from torchtext.data import Dataset
from torchtext.vocab import Vocab
from util import data_loader
import config
import torch


class RParsedTextLField(data.Field):
    def __init__(self, eos_token='<pad>', lower=False, include_lengths=True):
        super(RParsedTextLField, self).__init__(
            eos_token=eos_token, lower=lower, include_lengths=True, preprocessing=lambda parse: [
                t for t in parse if t not in ('(', ')')],
            postprocessing=lambda parse, _, __: [
                list(reversed(p)) for p in parse])


class ParsedTextLField(data.Field):
    def __init__(self, eos_token='<pad>', lower=False, include_lengths=True):
        super(ParsedTextLField, self).__init__(
            eos_token=eos_token, lower=lower, include_lengths=True, preprocessing=lambda parse: [
                t for t in parse if t not in ('(', ')')])

    def merge_vocab(self, *args, **kwargs):
            """Construct the Vocab object for this field from one or more datasets.

            Arguments:
                Positional arguments: Dataset objects or other iterable data
                    sources from which to construct the Vocab object that
                    represents the set of possible values for this field. If
                    a Dataset object is provided, all columns corresponding
                    to this field are used; individual columns can also be
                    provided directly.
                Remaining keyword arguments: Passed to the constructor of Vocab.
            """
            counter = Counter()
            sources = []
            for arg in args:
                # print(arg)
                if isinstance(arg, Dataset):
                    sources += [getattr(arg, name) for name, field in
                                arg.fields.items()
                                if field is self or
                                isinstance(field, SSUField)]
                else:
                    sources.append(arg)
            for data in sources:
                for x in data:
                    if not self.sequential:
                        x = [x]
                    counter.update(x)
            specials = list(OrderedDict.fromkeys(
                tok for tok in [self.pad_token, self.init_token, self.eos_token]
                if tok is not None))
            self.vocab = Vocab(counter, specials=specials, **kwargs)

    def plugin_new_words(self, new_vocab):
        for word, i in new_vocab.stoi.items():
            if word in self.vocab.stoi:
                continue
            else:
                self.vocab.itos.append(word)
                self.vocab.stoi[word] = len(self.vocab.itos) - 1


class SSTTextLField(data.Field):
    def __init__(self, tokenize=data.get_tokenizer('spacy'), eos_token='<pad>', lower=False, include_lengths=True):
        super(SSTTextLField, self).__init__(
            tokenize=tokenize,
            eos_token=eos_token, lower=lower, include_lengths=include_lengths)


class MNLI(data.ZipDataset, data.TabularDataset):
    # url = 'http://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    filename = 'multinli_0.9.zip'
    dirname = 'multinli_0.9'

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.premise), len(ex.hypothesis))

    @classmethod
    def splits(cls, text_field, label_field, parse_field=None, genre_field=None, root='.',
               train=None, validation=None, test=None):
        """Create dataset objects for splits of the SNLI dataset.
        This is the most flexible way to use the dataset.
        Arguments:
            text_field: The field that will be used for premise and hypothesis
                data.
            label_field: The field that will be used for label data.
            parse_field: The field that will be used for shift-reduce parser
                transitions, or None to not include them.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose snli_1.0
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.jsonl'.
            validation: The filename of the validation data, or None to not
                load the validation set. Default: 'dev.jsonl'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'test.jsonl'.
        """
        path = cls.download_or_unzip(root)
        if parse_field is None:
            return super(MNLI, cls).splits(
                os.path.join(path, 'multinli_0.9_'), train, validation, test,
                format='json', fields={'sentence1': ('premise', text_field),
                                       'sentence2': ('hypothesis', text_field),
                                       'gold_label': ('label', label_field)},
                filter_pred=lambda ex: ex.label != '-')
        return super(MNLI, cls).splits(
            os.path.join(path, 'multinli_0.9_'), train, validation, test,
            format='json', fields={'sentence1_binary_parse':
                                   [('premise', text_field),
                                    ('premise_transitions', parse_field)],
                                   'sentence2_binary_parse':
                                   [('hypothesis', text_field),
                                    ('hypothesis_transitions', parse_field)],
                                   'gold_label': ('label', label_field),
                                   'genre': ('genre', genre_field)},
            filter_pred=lambda ex: ex.label != '-')


if __name__ == "__main__":
    pass