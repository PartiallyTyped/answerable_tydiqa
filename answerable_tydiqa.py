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

import datasets


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
    language:str = "english"
    monolingual:bool = True
    task:Literal["cls", "qa"]= "none"
    def __init__(self, **kwargs):
        language = kwargs.pop("language", "all")
        monolingual = kwargs.pop("monolingual", True)
        task = kwargs.pop("task", "qa")
        super().__init__(**kwargs)
        self.language = language
        self.monolingual = monolingual
        self.task = task


VERSION = datasets.Version("1.1.0a")
RAW="raw"
PREPROCESSED="preprocessed"
PREPROC_FOR_CLASSIFICATION = "preprocessed for classification"
TOKENIZED_FOR_CLASSIFICATION = "tokenized for classification"

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
    # BUILDER_CONFIGS = [
        # TydiqaBuilderConfig(name=name, version=VERSION)
        # for language in ["english", "finnish", "japanese"]
        # for monolingual in [True, False]
        # for name in [RAW, PREPROCESSED, PREPROC_FOR_CLASSIFICATION, TOKENIZED_FOR_CLASSIFICATION]
    # ]
    BUILDER_CONFIGS = [
        # TydiqaBuilderConfig(name="raw", version=VERSION),
        TydiqaBuilderConfig(name="preprocessed", version=VERSION),
    ]

    DEFAULT_CONFIG_NAME = RAW # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        if self.config.name == PREPROCESSED and self.config.task=="qa":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "language": datasets.Value("string"),
                    "golds": datasets.features.Sequence(
                        {
                            "answer_text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            )
        elif self.config.name == PREPROCESSED and self.config.task == "cls":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "language": datasets.Value("string"),
                    "label": datasets.Value("bool"),
                }
            )
        else:
            raise ValueError("Unknown config name")
        
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
            RAW: RAW,
            PREPROCESSED: PREPROCESSED,
            PREPROC_FOR_CLASSIFICATION: "preprocessed_for_classification",
            TOKENIZED_FOR_CLASSIFICATION: "tokenized_for_classification"
        }[self.config.name]
        url = "https://raw.githubusercontent.com/PartiallyTyped/answerable_tydiqa/data/{split}/{name}.json"
        urls = {
            "train": url.format(split="train", name=name),
            "validation": url.format(split="validation", name=name),
        }
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
        else:
            check_language = lambda x: x != language

        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                language = data["language"]
                
                if not check_language(language):
                    continue
                
                if self.config.name == PREPROCESSED and self.config.task == "qa":
                    yield key, {
                        "id": data["id"],
                        "context": data["context"],
                        "question": data["question"],
                        "golds": data["golds"],
                        "language": language,
                    }
                elif self.config.name == PREPROCESSED and self.config.task == "cls":
                    yield key, {
                        "id": data["id"],
                        "context": data["context"],
                        "question": data["question"],
                        "label": any(s!=-1 for s in data["golds"]["answer_start"]),
                        "language":language
                    }
                else:
                    raise ValueError("Unknown config name")

