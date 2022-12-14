# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""

import csv
import json
import os
from collections import defaultdict
from typing import Literal
import pyarrow.parquet as pq
import datasets
from pathlib import Path

import spacy
from datasets import load_dataset
from xxhash import xxh128_hexdigest
from multiprocessing import cpu_count

from toolz import curry
import re
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_tags, strip_punctuation, strip_non_alphanum
from toolz import compose
from itertools import compress


strip_references = curry(re.sub)(r"\[\d+\]", "")
strip_quotes = curry(re.sub)(r"['\"]", "")
strip_double_quotes = curry(re.sub)(r"['\"]{2}", "")
strip_ellipsis = curry(re.sub)(r"\.{3}", "")
punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.()-"
remove_punc = curry(re.sub)(f"[{punc}]", "")

preprocess_string = compose(
    strip_multiple_whitespaces,
    strip_tags,
    strip_punctuation,
    strip_references,
    strip_double_quotes,
    strip_ellipsis,
    remove_punc,
)

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""


# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)

class TydiqaBuilderConfig(datasets.BuilderConfig):
    """BuilderConfig for AnswerableTydiqa"""
    language: str = "english"
    monolingual: bool = True
    split_to_sentences: bool = True

    def __init__(self, **kwargs):
        self.language = kwargs.pop("language", "all")
        self.monolingual = kwargs.pop("monolingual", True)
        self.split_to_sentences = kwargs.pop("split_to_sentences", True)
        super().__init__(**kwargs)

    def language_filter(self, split, language):
        if language == "all":
            return True

        if self.monolingual:
            return self.language == language

        # multilingual
        if split == "train":
            return self.language == language
        return self.language != language

    def urls(self):
        raise NotImplementedError()


class RawConfig(TydiqaBuilderConfig):

    def __init__(self, **kwargs):
        super().__init__(name="raw", **kwargs)

    @property
    def features(self):
        return datasets.Features(
            {
                "seq_id": datasets.Value("string"),
                "context": datasets.Value("string"),
                "question": datasets.Value("string"),
                "golds": datasets.features.Sequence(
                    {
                        "answer_text": datasets.Value("string"),
                        "answer_start": datasets.Value("int32"),
                    }
                ),
                "language": datasets.Value("string"),
            }
        )

    @property
    def urls(self):
        return {}

    def load_dataset(self, split):
        ds = load_dataset("copenlu/answerable_tydiqa", split=split)

        tokenizers = {
            "english": "en_core_web_sm",
            "finnish": "fi_core_news_sm",
            "japanese": "ja_core_news_sm",
        }
        tokenizers = {lang: spacy.load(model) for lang, model in tokenizers.items()}

        ds = (ds
              .filter({"english", "finnish", "japanese"}.__contains__, input_columns=["language"], num_proc=cpu_count())
              .rename_columns({"question_text": "question", "document_plaintext": "context"})
              .remove_columns(["document_url", "document_title"])
              .map(lambda example: {"seq_id": xxh128_hexdigest(example["context"] + example["question"])}, num_proc=cpu_count())
        )
        if self.split_to_sentences:
            ds = ds.map(self.separate_sentences, batched=True, batch_size=1, fn_kwargs={"tokenizers": tokenizers}, remove_columns=["annotations"], num_proc=cpu_count())
        else:
            ds = ds.rename_columns({"annotations": "golds"})
        if self.language == "all":
            languages = {"english", "finnish", "japanese"}
        else:
            languages = {self.language}
        ds = ds.filter(languages.__contains__, input_columns=["language"], num_proc=cpu_count())
        return ds

    @staticmethod
    def separate_sentences(example, tokenizers):
        question = example["question"][0]
        language = example["language"][0]
        annotations = example["annotations"][0]
        context = example["context"][0]
        seq_id = example["seq_id"][0]


        tokenizer = tokenizers[language]

        answers = [
            (sent, answer_start + sent.start_char, answer_start + sent.end_char)
            for answer_text, answer_start in zip(annotations["answer_text"], annotations["answer_start"])
            for sent in tokenizer(answer_text).sents
        ]
        out = defaultdict(list)
        for sentence in tokenizer(context).sents:
            out["language"].append(language)
            out["context"].append(sentence.text)
            out["seq_id"].append(seq_id)
            out["question"].append(question)

            sentence_start = sentence.start_char
            sentence_end = sentence.end_char

            out["golds"].append(defaultdict(list))

            for (answer_text, answer_start, answer_end) in answers:
                if sentence_start <= answer_start < sentence_end:
                    termination = min(answer_end, sentence_end) - sentence_start
                    beginning = answer_start - sentence_start
                    out["golds"][-1]["answer_text"].append(sentence.text[beginning:termination])
                    out["golds"][-1]["answer_start"].append(beginning)
            if not out["golds"][-1]:
                out["golds"][-1]["answer_text"].append('')
                out["golds"][-1]["answer_start"].append(-1)

        return out

    @staticmethod
    def extract(example):
        return example

class TokenizedConfig(TydiqaBuilderConfig):
    preprocess: bool = False
    def __init__(self, preprocess=False, **kwargs):
        super().__init__(name="tokenized", **kwargs)
        self.preprocess = preprocess
    
    @property
    def features(self):

        base = {
                "seq_id": datasets.Value("string"),
                "context": datasets.features.Sequence(datasets.Value("string")),
                "question": datasets.features.Sequence(datasets.Value("string")),
                "golds": datasets.features.Sequence(
                    {
                        "answer_text": datasets.features.Sequence(datasets.Value("string")),
                        "answer_start": datasets.Value("int32"),
                    }
                ),
                "language": datasets.Value("string"),
            }
        task = getattr(self, "task", None)
        if task == "qa":
            base["label"] = datasets.features.Sequence(datasets.Value("int32"))
        if task == "cls":
            base["label"] = datasets.Value("int32")

        return datasets.Features(base)

    @property
    def urls(self):
        return {}
    
    def load_dataset(self, split):
        ds = load_dataset("PartiallyTyped/answerable_tydiqa", split=split, language=self.language, monolingual=self.monolingual, split_to_sentences=self.split_to_sentences)
       # first tokenize question, context using spacy
        # then preprocess using gensim
        # then remove empty tokens
        # use answer_start and len(answer_text) to find the span of tokens in the context

        tokenizers = {
            "english": "en_core_web_sm",
            "finnish": "fi_core_news_sm",
            "japanese": "ja_core_news_sm",
        }
        tokenizers = {lang: spacy.load(model) for lang, model in tokenizers.items()}
        ds = ds.map(self.tokenize, fn_kwargs={"tokenizers": tokenizers}, num_proc=cpu_count())
        if self.preprocess:
            ds = ds.map(self.preprocess, num_proc=cpu_count())
        ds = ds.map(self.remove_empty, num_proc=cpu_count())

    def tokenize(self, example, tokenizers):
        context = example["context"]
        question = example["question"]
        language = example["language"]
        seq_id = example["seq_id"]
        tokenizer = tokenizers[language]
        context, token_start, token_end = zip(*[(token.text, token.idx, token.idx + len(token.text)) for token in tokenizer(context)])
        question = [token.text for token in tokenizer(question)]
        labels = [0] * len(context)

        for gold in example["golds"]:
            answer_text = gold["answer_text"]
            answer_start = gold["answer_start"]
            answer_end = answer_start + len(answer_text)
            for i,(token_start, token_end) in enumerate(zip(token_start, token_end)):
                if answer_start<=token_start<answer_end:
                   labels[i] = 1

        return {
            "seq_id": seq_id,
            "context": context,
            "question": question,
            "golds": example["golds"],
            "language": language,
            "label": labels,
        }

    def preprocess(self, example):
        context = example["context"]
        question = example["question"]
        language = example["language"]
        seq_id = example["seq_id"]
        labels = example["label"]

        context = preprocess_string(context, language)
        question = preprocess_string(question, language)

        return {
            "seq_id": seq_id,
            "context": context,
            "question": question,
            "golds": example["golds"],
            "language": language,
            "label": labels,
        }
    
    def remove_empty(self, example):
        context = example["context"]
        question = example["question"]
        labels = example["label"]
        selectors = list(map(bool, context))
        context = list(compress(context, selectors))
        labels = list(compress(labels, selectors))

        return {
            "context": context,
            "question": question,
            "golds": example["golds"],
            "language": example["language"],
            "label": labels,
            "seq_id": example["seq_id"],
        }

class AnswerableTydiqa(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    BUILDER_CONFIG_CLASS = TydiqaBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        # TydiqaBuilderConfig(name="raw", version=VERSION),
        # TydiqaBuilderConfig(name=PREPROCESSED, version=VERSION),
        # TydiqaBuilderConfig(name=TOKENIZED, version=VERSION),
        # TydiqaBuilderConfig(name=BPEMB, version=VERSION),
        # TydiqaBuilderConfig(name=HASHINGTRICK, version=VERSION),
        # TydiqaBuilderConfig(name=HASHINGTRICK_BPEMB, version=VERSION),
        RawConfig(),
        TokenizedConfig(),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=self.config.features,
            # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "split": "validation",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, split):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        dataset = self.config.load_dataset(split)
        for i, row in enumerate(dataset):
            yield i, self.config.extract(row)


def extract_hashingtrick(data):
    return {
        "id": data["id"],
        "language": data["language"],
        "label": data["label"],
        "embeddings": data["embeddings"],
    }


def extract_hashingtrick_bpemb(data):
    return {
        "id": data["id"],
        "language": data["language"],
        "label": data["label"],
        "embeddings": data["embeddings"],
        "context": data["context"],
        "question": data["question"],
    }


def extract_preprocessed(data):
    return {
        "id": data["id"],
        "context": data["context"],
        "question": data["question"],
        "golds": data["golds"],
        "language": data["language"],
        "label": any(s != -1 for s in data["golds"]["answer_start"]),
    }


def extract_tokenized(data):
    return {
        "id": data["id"],
        "iob_label": data["iob_label"],
        "context": data["context"],
        "question": data["question"],
        "golds": data["golds"],
        "language": data["language"],
        "cls_label": any(data["iob_label"]),
    }
