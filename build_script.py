import datasets as D
from joblib import cpu_count
import spacy
from absl import flags, app
from spacy.lang.en import English
from spacy.lang.fi import Finnish
from spacy.lang.ja import Japanese

from toolz import curry, compose
import re
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_tags, strip_punctuation
from collections import defaultdict
from xxhash import xxh64_hexdigest
import pathlib as pl
import nltk

strip_references = curry(re.sub)(r"\[\d+\]", "")
strip_quotes = curry(re.sub)(r"['\"]", "")
strip_double_quotes = curry(re.sub)(r"['\"]{2}", "")
strip_ellipsis = curry(re.sub)(r"\.{3}", "")

preprocess_fn = compose(
    strip_multiple_whitespaces,
    strip_tags,
    strip_references,
    strip_quotes,
    strip_double_quotes,
    strip_ellipsis,
)

punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
preprocess_fn2 = compose(strip_multiple_whitespaces,strip_tags,str.lower, lambda x: re.sub(r"[%s]+" % punc, "", x))
preprocess_all = compose(
    str.strip,
    preprocess_fn,
    strip_multiple_whitespaces,
    strip_tags,
    str.lower,
    curry(re.sub, r"[%s]+" % punc, ""),
)

MODE = flags.DEFINE_enum("mode", None, ["preprocessed", "tokenized", "bpemb"], "Which data to build")

def build_preprocessed():
    processors = {
        "english": English(),
        "finnish": Finnish(),
        "japanese": Japanese(),
    }
    for v in processors.values():
        v.add_pipe("sentencizer")

    raw = D.load_dataset("copenlu/answerable_tydiqa").filter(processors.__contains__, input_columns="language")

    def transform(example):
        context = preprocess_fn(example["document_plaintext"][0])
        question = preprocess_fn(example["question_text"][0])
        golds = example["annotations"][0]
        language = example["language"][0]
        answer = preprocess_fn(golds["answer_text"][0])
        seq_id = xxh64_hexdigest(context + question)
        
        nlp = processors[language]
        answers = list(map(str, nlp(answer).sents))
        answers = [preprocess_all(a) for a in answers]

        sentences = list(map(str, nlp(context).sents))
        sentences = [preprocess_all(s) for s in sentences]

        question = preprocess_all(question)
        
        passes_test = True
        if bool(answer):
            passes_test = any(answer in sent for sent in sentences for answer in answers)
        
        return {
            "id": [seq_id],
            "context": [sentences],
            "question": [question],
            "language": [language],
            "answers": [answers],
            "passes_test": [passes_test],
        }
    
    def distribute(sample):
        context = sample["context"][0]
        question = sample["question"][0]
        language = sample["language"][0]
        answers = sample["answers"][0]
        seq_id = sample["id"][0]
        out = defaultdict(list)
        for sent in context:
            for answer in answers:
                if answer in sent:
                    out["id"].append(seq_id)
                    out["context"].append(sent)
                    out["question"].append(question)
                    out["language"].append(language)
                    out["golds"].append({"answer_text": [answer], "answer_start": [sent.index(answer)]})
                    break
            else:
                out["id"].append(seq_id)
                out["context"].append(sent)
                out["question"].append(question)
                out["language"].append(language)
                out["golds"].append({"answer_text": [""], "answer_start": [-1]})

        return out


    
    out = (
        raw.map(transform, batched=True, batch_size=1, remove_columns=["document_plaintext", "question_text", "annotations", "language", "document_title", "document_url"])
        .filter(bool, input_columns="passes_test")
    )
    print(raw)
    print(out)
    distributed = out.map(distribute, batched=True, batch_size=1, remove_columns=["context", "question", "language", "answers", "passes_test"])
    print(distributed)
    for key, value in distributed.items():
        path = pl.Path(f"{key}/preprocessed.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        value.to_json(path)

    return out


def build_tokenized():
    # use spacy large models
    tokenizers = {
        "english": spacy.load("en_core_web_lg"),
        "finnish": spacy.load("fi_core_news_lg"),
        "japanese": spacy.load("ja_core_news_lg"),
    }
    def tokenize(example):
        context = example["context"]
        question = example["question"]
        language = example["language"]
        golds = example["golds"]
        nlp = tokenizers[language]
        ctx = nlp(context)
        tokens = [t.text for t in ctx]
        token_starts = [t.idx for t in ctx]
        token_ends = [t.idx + len(t.text) for t in ctx]

        q = nlp(question)
        q_tokens = [t.text for t in q]

        answer_start = golds["answer_start"][0]
        answer_end = answer_start + len(golds["answer_text"][0])
        iob = [0] * len(tokens)
        for i, (start, end) in enumerate(zip(token_starts, token_ends)):
            if start >= answer_start and  end <= answer_end:
                iob[i] = 1
        
        return {
            "iob_label": iob,
            "context": tokens,
            "question": q_tokens,
            "language": language,
            "id": example["id"],
            "golds": golds,
        }
    ds = D.load_dataset("PartiallyTyped/answerable_tydiqa", "preprocessed")
    ds = ds.map(tokenize)

    for key, value in ds.items():
        path = pl.Path(f"{key}/tokenized.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        value.to_json(path)

def build_bpemb():
    from bpemb import BPEmb
    tokenized = D.load_dataset("PartiallyTyped/answerable_tydiqa", "tokenized")
    encoders = {
        "english": BPEmb(lang="en", vs=100000, dim=300),
        "finnish": BPEmb(lang="fi", vs=100000, dim=300),
        "japanese": BPEmb(lang="ja", vs=100000, dim=300),
    }
    def encode(example):
        context = example["context"]
        question = example["question"]
        language = example["language"]
        golds = example["golds"]
        encoder = encoders[language]
        ctx = encoder.encode_ids(context)
        q = encoder.encode_ids(question)
        iob_labels = example["iob_label"]
        iob_labels = [[l]*len(t) for l, t in zip(iob_labels, ctx)]
        iob_labels = sum(iob_labels, [])
        context = sum(ctx, [])
        question = sum(q, [])

        return {
            "iob_label": iob_labels,
            "context": context,
            "question": question,
            "language": language,
            "id": example["id"],
            "golds": golds,
        }
    ds = tokenized.map(encode)
    for key, value in ds.items():
        path = pl.Path(f"{key}/bpemb.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        value.to_json(path)
    

def main(_):
    if MODE.value == "preprocessed":
        build_preprocessed()
    elif MODE.value == "tokenized":
        build_tokenized()
    elif MODE.value == "bpemb":
        build_bpemb()
    else:
        raise ValueError("Not implemented yet")
    

if __name__ == "__main__":
    flags.mark_flags_as_required(["mode"])
    app.run(main)