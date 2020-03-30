from readXls import ReadXls
from tagger import Tagger
from pymagnitude import Magnitude
import _pickle as pickle
from os import path, mkdir
from re import sub

use_pickled_data = True
try:
    mkdir("pickle")
except:
    pass

if __name__ == "__main__":
    xls = ReadXls("data/AllDiscussionData.xls")
    messages = xls.get_column_with_name("Message")
    print(messages)

    if not path.exists("pickle/lemmas.pkl") or not use_pickled_data:
        tagger = Tagger()
        # remove all non alphanumeric characters
        no_non_alphanumeric_chars = map(lambda t: sub(r'[^a-zA-Z0-9]+', ' ', str(t)), messages)
        # if a letter is repeated 3 times its most likely to emphisize the text so its shortened to 1 repetition
        no_triple_chars = map(lambda t: sub(r'(\w)\1\1*', r'\1', str(t)), no_non_alphanumeric_chars)
        # get lemmas change all lemmas to lower case
        lemmas = map(lambda t: list(map(lambda u: u.lower(), tagger.lemmatiser(t, tagging=True)[1])), no_triple_chars)
        # if lemmas are empty mark it with _
        lemmas = list(map(lambda t: ['_'] if (not t) else t, lemmas))
        if use_pickled_data:
            pickle.dump(lemmas, open("pickle/lemmas.pkl", "wb"))
    if use_pickled_data:
        lemmas = pickle.load(open("pickle/lemmas.pkl", "rb"))
    print(lemmas)

    if not path.exists("pickle/wikiWV.pkl") or not use_pickled_data:
        wv_embeddings = Magnitude("embeddings/wiki.sl.magnitude")
        word_vec = list(map(lambda t: wv_embeddings.query(t), lemmas))
        if use_pickled_data:
            pickle.dump(word_vec, open("pickle/wikiWV.pkl", "wb"))
    if use_pickled_data:
        word_vec = pickle.load(open("pickle/wikiWV.pkl", "rb"))
    print(word_vec)

    if not path.exists("pickle/elmoWV.pkl") or not use_pickled_data:
        wv_embeddings = Magnitude("embeddings/slovenian-elmo.weights.magnitude")
        word_vec = list(map(lambda t: list(wv_embeddings.query(t)), lemmas))
        if use_pickled_data:
            pickle.dump(word_vec, open("pickle/elmoWV.pkl", "wb"))
    if use_pickled_data:
        word_vec = pickle.load(open("pickle/elmoWV.pkl", "rb"))
    print(word_vec)
