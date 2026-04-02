import spacy
from spacy.language import Language
from typing import List, Dict, Any
from spacy.util import registry, compile_suffix_regex
from spacy.tokenizer import Tokenizer
import unicodedata
import re
import string
import numpy as np
from spacy.lookups import load_lookups, Lookups
from spacy.tokens import Token, Doc




# ========== que_exceptions ========== #
que_exceptions = [
    # quisque / quique
    "quisque", "quidque", "quicque", "quodque", "cuiusque", "cuique", "quemque", "quamque",
    "quoque", "quaque", "quique", "quaeque", "quorumque", "quarumque", "quibusque", "quosque", "quasque",
    # uterque
    "uterque", "utraque", "utrumque", "utriusque", "utrique", "utrumque", "utramque", "utroque",
    "utraque", "utrique", "utraeque", "utrorumque", "utrarumque", "utrisque", "utrosque", "utrasque",
    # 其他例外（保持原样）
    "absque", "abusque", "adaeque", "adusque", "aeque", "antique", "atque", "circumundique",
    "conseque", "cumque", "cunque", "denique", "deque", "donique", "hucusque", "inique", "inseque",
    "itaque", "longinque", "namque", "neque", "oblique", "peraeque", "praecoque", "propinque",
    "qualiscumque", "quandocumque", "quandoque", "quantuluscumque", "quantumcumque",
    "quantuscumque", "quinque", "quocumque", "quomodocumque", "quomque", "quotacumque",
    "quotcumque", "quotienscumque", "quotiensque", "quotusquisque", "quousque", "relinque",
    "simulatque", "torque", "ubicumque", "ubique", "undecumque", "undique", "usque",
    "usquequaque", "utcumque", "utercumque", "utique", "utrimque", "utrique", "utriusque",
    "utrobique", "utrubique"
]

# ========== lookup_lemmatizer ========== #
blank_nlp = spacy.blank("la")
lookups_data = load_lookups(lang=blank_nlp.vocab.lang, tables=["lemma_lookup"])
LOOKUPS = lookups_data.get_table("lemma_lookup")
Token.set_extension("predicted_lemma", default=None, force=True)

@Language.component("lookup_lemmatizer")
def lookup_lemmatizer(doc):
    for token in doc:
        token._.predicted_lemma = token.lemma_
        if token.text in string.punctuation:
            token.lemma_ = token.text
            token.pos_ = "PUNCT"
            token.tag_ = "punc"
        if token.text == "que" and (token.pos_ == "CCONJ" or token.tag_ == "conjunction"):
            token.lemma_ = token.text
        token.lemma_ = LOOKUPS.get(token.text, token.lemma_)
        if token.text[0].isupper() and token.text not in LOOKUPS:
            token.lemma_ = LOOKUPS.get(token.text.lower(), token.lemma_)
    return doc

# ========== trf_vectors ========== #
@Language.factory("trf_vectors")
class TrfContextualVectors:
    def __init__(self, nlp: Language, name: str):
        self.name = name
        Doc.set_extension("trf_token_vecs", default=None)

    def __call__(self, doc):
        if isinstance(doc, str):
            doc = self.nlp(doc)
        vec_idx_splits = np.cumsum(doc._.trf_data.align.lengths)
        trf_vecs = doc._.trf_data.tensors[0].reshape(-1, 768)
        vec_idxs = np.split(doc._.trf_data.align.dataXd, vec_idx_splits)
        vecs = np.stack([trf_vecs[idx].sum(0) for idx in vec_idxs[:-1]])
        doc._.trf_token_vecs = vecs
        doc.user_token_hooks["vector"] = self.vector
        doc.user_token_hooks["has_vector"] = self.has_vector
        return doc

    def vector(self, token):
        return token.doc._.trf_token_vecs[token.i]

    def has_vector(self, token):
        return True

# ========== normer ========== #
@Language.component("normer")
def normer(doc):
    def norm(text):
        return text.replace("v", "u").replace("j", "i").replace("V", "U").replace("J", "I")
    for token in doc:
        token.norm_ = norm(token.norm_)
    return doc

# ========== remorpher ========== #
Token.set_extension("remorph", default=None, force=True)

@Language.component("remorpher")
def remorpher(doc):
    for token in doc:
        token._.remorph = token.morph
        morph = token.morph.to_dict()
        if morph.get("Tense"):
            if morph["Tense"] in {"Perf", "Imp"}:
                morph["Tense"] = "Past"
            elif morph["Tense"] == "FutPerf":
                morph["Tense"] = "Fut"
        token.set_morph(morph)
    return doc

@registry.tokenizers("latin_core_tokenizer")
def create_latin_tokenizer():
    def create_tokenizer(nlp):
        tokenizer = LatinTokenizer(nlp.vocab)
        suffixes = nlp.Defaults.suffixes + ["que", "qve"]
        suffix_regex = compile_suffix_regex(suffixes)
        tokenizer.suffix_search = suffix_regex.search
        for item in que_exceptions:
            for form in [item, item.lower(), item.title(), item.upper()]:
                tokenizer.add_special_case(form, [{"ORTH": form}])
        return tokenizer
    return create_tokenizer

class LatinTokenizer(Tokenizer):
    def separate_ligatures(self, text: str) -> str:
        return text.replace("Æ", "Ae").replace("Œ", "Oe").replace("æ", "ae").replace("œ", "oe")

    def remove_macrons(self, text: str) -> str:
        macron_map = str.maketrans("āēīōūȳĀĒĪŌŪȲ", "aeiouyAEIOUY")
        return text.translate(macron_map)

    def remove_accents(self, text: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

    def norm_spacing(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def preprocess(self, text: str) -> str:
        text = self.separate_ligatures(text)
        text = self.remove_macrons(text)
        text = self.remove_accents(text)
        text = self.norm_spacing(text)
        return text

    def __call__(self, text):
        processed_text = self.preprocess(text)
        return super().__call__(processed_text)
