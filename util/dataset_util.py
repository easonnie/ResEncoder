from collections import Counter, OrderedDict

from torchtext import data
import torchtext
import os
import torch
import six
import config


class SST_UTF8(data.Dataset):
    def __init__(self, path, text_field, label_field, subtrees=False,
                 fine_grained=False, binary=True, **kwargs):
        """Create an SST dataset instance given a path and fields.

        Arguments:
            path: Path to the data file.
            text_field: The field that will be used for text data.
            newline_eos: Whether to add an <eos> token for every newline in the
                data file. Default: True.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        fields = [('text', text_field), ('label', label_field)]

        def get_label_str(label):
            pre = 'very ' if fine_grained else ''
            return {'0': pre + 'negative', '1': 'negative', '2': 'neutral',
                    '3': 'positive', '4': pre + 'positive', None: None}[label]

        label_field.preprocessing = data.Pipeline(get_label_str)

        if binary:
            filter_pred = lambda ex: ex.label != 'neutral'
        else:
            filter_pred = None

        with open(os.path.expanduser(path), encoding='utf-8') as f:
            if subtrees:
                examples = [ex for line in f for ex in
                            data.Example.fromtree(line, fields, True)]
            else:
                examples = [data.Example.fromtree(line, fields) for line in f]
        super(SST_UTF8, self).__init__(examples=examples, fields=fields, filter_pred=filter_pred, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='.',
               train='/train.txt', validation='/dev.txt', test='/test.txt',
               train_subtrees=False, **kwargs):
        path = os.path.join(root, 'SST')

        train_data = None if train is None else cls(
            path + train, text_field, label_field, subtrees=train_subtrees,
            **kwargs)
        val_data = None if validation is None else cls(
            path + validation, text_field, label_field, **kwargs)
        test_data = None if test is None else cls(
            path + test, text_field, label_field, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


class TabularUTF8Dataset(data.Dataset):
    def __init__(self, path, format, fields, **kwargs):

        make_example = {
            'json': data.Example.fromJSON, 'dict': data.Example.fromdict,
            'tsv': data.Example.fromTSV, 'csv': data.Example.fromCSV}[format.lower()]

        with open(os.path.expanduser(path), encoding='utf-8') as f:
            examples = [
                make_example(line.decode('utf-8') if six.PY2 else line, fields)
                for line in f]

        if make_example in (data.Example.fromdict, data.Example.fromJSON):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(TabularUTF8Dataset, self).__init__(examples, fields, **kwargs)


class SSUField(data.Field):
    def __init__(self, tokenize=data.get_tokenizer('spacy'), eos_token='<pad>', include_lengths=True):
        super(SSUField, self).__init__(tokenize=tokenize,
                                       eos_token=eos_token,
                                       include_lengths=include_lengths)

    def merge_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            # print(arg)
            if isinstance(arg, torchtext.data.Dataset):
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
        self.vocab = torchtext.data.Vocab(counter, specials=specials, **kwargs)