import json
from typing import List
from nltk.corpus import wordnet
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from flair.models import SequenceTagger

# Stopwords list. We know that libraries like spicy or nltk have their own lists
# of stop words but we prefered doing it in this way in line with the work
# developed by https://hal.archives-ouvertes.fr/hal-02933266
stop_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', ''
]

# Similar to the stopwords, these lists were created by
# https://hal.archives-ouvertes.fr/hal-02933266 and adapted to use json files
# instead of the original ones for developing reasons.

# List of speculative adverbs such as `absolutely`, `doubtlessly`, among others.
with open("./words_lists/speculative.json", "rb") as f:
    speculative_list = json.load(f)

# List of scalar adverbs such as `barely`, `profounfly`, among others.
with open("./words_lists/scalarity.json", "rb") as f:
    scalarity_list = json.load(f)

# Dictionary of words and possible replacements. For example: "supremacy":
# ["dominance", "superiority"].
with open("./words_lists/special_tokens.json", "rb") as f:
    special_tokens = json.load(f)

# Dictionary of words for each NEs (named entities): PER, LOC, ORG, and MISC.
with open("./words_lists/entities_list.json", "rb") as f:
    entities_list = json.load(f)

# Flair taggers and embeddings.
tagger_pos = SequenceTagger.load("pos")
tagger_ner = SequenceTagger.load("ner")
fasttext_emb = WordEmbeddings("en")
fasttext_doc_emb = DocumentPoolEmbeddings([fasttext_emb])


def get_synonyms(word: str,
                 pos: str = None,
                 only_hyponyms: bool = False) -> List[str]:
    """
    Returns a list of synonyms for `word`. If `only_hyponyms` returns a list of
    hyponyms. Note that, if `word` has a certain part of speech P, the resultant
    synonyms might not belong to P. To keep only synonyms of a certain part of
    speech use `pos`.

    parameters:
        - word (str) word to find the synonyms or hyponyms.
        - pos (str) to get synoynms or hyponyms of a certain Part of Speech. The
          values for this parameter are:
            `n` noun, also tagged as `NN` by other pos taggers. `a` adjectives,
            also tagged as `JJ` by other pos taggers. `s` adjectives, also
            tagged as `JJ` by other pos taggers. `r` adverbs, also tagged as
            `RB` by other pos taggers. `v` verbs, also tagged as `VB` by other
            pos taggers.
          if `pos` is not one of the values mentioned above, `get_synonyms` will
          return the empty list. if `pos` is None will `get_synonyms` will
          return synonyms or hyponyms belonging to any pos type.
        - only_hyponyms (bool) flag to return only hyponyms instead of synonyms.

    returns:
        synonyms (List[str]) list of synonyms or hyponyms.
    """
    synonyms = set()

    # Possible values for `pos`
    pos_types = ["n", "a", "s", "r", "v"]

    for syn in wordnet.synsets(word):
        # If pos is None this if is always true and includes all the candidates
        # returned by wordnet. Otherwise, it checks pos types.
        if (pos is None) or ((syn.pos() in pos_types) and (syn.pos() == pos)):
            for l in syn.hyponyms() if only_hyponyms else syn.lemmas():

                # Hyphens and underscores are replaced by spaces.
                synonym = l.name().replace("_", " ").replace("-", " ").lower()

                # Keep only utf-8 chars.
                synonym = "".join([
                    char for char in synonym
                    if char in ' qwertyuiopasdfghjklzxcvbnm'
                ])

                # Add synonym or hyponym
                synonyms.add(synonym)

    # Remove the word introduced as input if wordnet suggested it as synonym or
    # hyponym.
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def replace_exp_to_sent(sent: str, exp: str, start_pos: int,
                        end_pos: int) -> str:
    """
    Given a sentence `sent`, an expression `exp`, and offsets `start_pos`,
    `end_pos`, the function replaces the expression of `sent` that is in the
    range [start_pos, end_pos] with `exp`.

    parameters:
        - `sent` (str) sentence to expand.
        - `exp` (str) expression to add.
        - `start_pos` (int) index of `sent`.
        - `end_pos` (int) index of `sent`.

    returns:
        example (str) modification of `sent` with the replacement of `exp`.
    """
    return sent[:start_pos] + exp + sent[end_pos:]
