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
from typing import Literal
import pyarrow.parquet as pq
import datasets
from pathlib import Path
from transformers import AutoTokenizer
from toolz import curry

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

VERSION = datasets.Version("2.0.1")
RAW="raw"
PREPROCESSED="preprocessed"
TOKENIZED = "tokenized"
BPEMB = "bpemb"
HASHINGTRICK = "hashingtrick"
HASHINGTRICK_BPEMB = "hashingtrick_bpemb"
TRANSFORMERS = "transformers"

class TydiqaBuilderConfig(datasets.BuilderConfig):
    """BuilderConfig for AnswerableTydiqa"""
    language:str = "english"
    monolingual:bool = True
    def __init__(self, **kwargs):
        language = kwargs.pop("language", "all")
        monolingual = kwargs.pop("monolingual", True)
        super().__init__(**kwargs)
        self.language = language
        self.monolingual = monolingual

class TydiqaTransformerConfig(TydiqaBuilderConfig):
    def __init__(self,
        source_dataset:Literal["raw", "preprocessed"]=RAW, tokenizer="xlm-roberta-base", pad_token=None, cls_token=None, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.padding_token = pad_token
        self.cls_token = cls_token
        self.source_dataset = source_dataset

COMMON_FEATURES = {
    "id": datasets.Value("string"),
    "language": datasets.Value("string"),
    "golds": datasets.features.Sequence(
                        {
                            "answer_text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                ),
}
FEATURES = {
    RAW: datasets.Features({
        **COMMON_FEATURES,
        "context": datasets.Value("string"),
        "question": datasets.Value("string"),
    }),
    BPEMB: datasets.Features({
                "iob_label": datasets.features.Sequence(datasets.Value("int32")),
                "cls_label": datasets.Value("bool"),
                "context": datasets.features.Sequence(datasets.Value("int32")),
                "question": datasets.features.Sequence(datasets.Value("int32")),
                **COMMON_FEATURES,
    }),
    TOKENIZED: datasets.Features(
                {
                    "iob_label": datasets.features.Sequence(datasets.Value("int32")),
                    "cls_label": datasets.Value("bool"),
                    "context": datasets.features.Sequence(datasets.Value("string")),
                    "question": datasets.features.Sequence(datasets.Value("string")),
                    **COMMON_FEATURES,
                }
            ),
    PREPROCESSED: datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "label": datasets.Value("bool"),
                    **COMMON_FEATURES,
                }
            ),
    HASHINGTRICK: datasets.Features(
        {
            "embeddings": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("float32"))),
            "label": datasets.Value("bool"),
            "id": datasets.Value("string"),
            "language": datasets.Value("string"),
        }
    ),
    HASHINGTRICK_BPEMB: datasets.Features(
        {
            "embeddings": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("float32"))),
            "context": datasets.features.Sequence(datasets.Value("int32")),
            "question": datasets.features.Sequence(datasets.Value("int32")),
            "label": datasets.Value("bool"),
            "id": datasets.Value("string"),
            "language": datasets.Value("string"),
        }
    ),
    TRANSFORMERS: datasets.Features(
        {
            "context_ids": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("int32"))),
            "question_ids": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("int32"))),
            "iob_label": datasets.features.Sequence(datasets.Value("int32")),
            "token_type_ids": datasets.features.Sequence(datasets.Value("int32")),
            "cls_label": datasets.Value("bool"),
            **COMMON_FEATURES,
        }
    )
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
        TydiqaBuilderConfig(name=PREPROCESSED, version=VERSION),
        TydiqaBuilderConfig(name=TOKENIZED, version=VERSION),
        TydiqaBuilderConfig(name=BPEMB, version=VERSION),
        TydiqaBuilderConfig(name=HASHINGTRICK, version=VERSION),
        TydiqaBuilderConfig(name=HASHINGTRICK_BPEMB, version=VERSION),
    ]

    DEFAULT_CONFIG_NAME = PREPROCESSED # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        features = FEATURES[self.config.name]
        
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features, # Here we define them above because they are different between the two configurations
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
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        name = {
            PREPROCESSED: PREPROCESSED,
            TOKENIZED: TOKENIZED,
            BPEMB: BPEMB,
            HASHINGTRICK: HASHINGTRICK,
            HASHINGTRICK_BPEMB: HASHINGTRICK_BPEMB,
            TRANSFORMERS: getattr(self.config, "source_dataset", "raw")
        }[self.config.name]

        url = "https://raw.githubusercontent.com/PartiallyTyped/answerable_tydiqa/data/{split}/{name}.pq"
        urls = {
            "train": url.format(split="train", name=name),
            "validation": url.format(split="validation", name=name),
        }
        print(urls)
        data_dir = dl_manager.download_and_extract(urls)
     
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
                    "language": self.config.language,
                    "monolingual": self.config.monolingual,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["validation"],
                    "split": "validation",
                    "language": self.config.language,
                    "monolingual": self.config.monolingual,
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split, language, monolingual):
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        if language == "all":
            check_language = lambda x: True
        elif monolingual or split=="train":
            check_language = language.__eq__
        elif not monolingual and split=="validation":
            s = {"finnish", "english", "japanese"}
            s.remove(language)
            check_language = s.__contains__
        # if the extension is parquet

        ds = datasets.Dataset(pq.read_table(filepath, memory_map=True))

        if self.config.name == TRANSFORMERS:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
            if self.config.pad_token is not None:
                tokenizer.add_special_tokens({"pad_token": self.config.pad_token})
            if self.config.cls_token is not None:
                tokenizer.add_special_tokens({"cls_token": self.config.cls_token})
            
            extract_transformer = extract_transformer_(tokenizer)

        
        for i, data in enumerate(ds):
            if not check_language(data["language"]):
                continue
            if self.config.name == PREPROCESSED:
                yield i, extract_preprocessed(data)
            elif self.config.name in (TOKENIZED, BPEMB):
                yield i, extract_tokenized(data)
            elif self.config.name == HASHINGTRICK:
                yield i, extract_hashingtrick(data)
            elif self.config.name == HASHINGTRICK_BPEMB:
                yield i, extract_hashingtrick_bpemb(data)
            elif self.config.name == TRANSFORMERS:
                yield i, extract_transformer(data)
            else:
                raise ValueError("Unknown config name")

@curry
def extract_transformer_(tokenizer, data):
    context = data["context"]
    question = data["question"]
    golds = data["golds"]
    answer_start = golds["answer_start"][0]
    answer = golds["answer_text"][0]
    answer_end = answer_start + len(answer)

    encodings = tokenizer(
        context,
        question,
        return_tensors="np",
        return_offsets_mapping=True,
        return_token_type_ids=True,
        return_attention_mask=False,
    )
    offset_mapping = encodings.pop("offset_mapping")
    token_type_ids = encodings.pop("token_type_ids")
    input_ids = encodings["input_ids"]

    # context_token_mask = 



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
            "label": any(s!=-1 for s in data["golds"]["answer_start"]),
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