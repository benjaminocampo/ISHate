from nltk import word_tokenize, pos_tag
from typing import List
from ISHate.augmentation.utils import get_synonyms


def ra(sent: str,
       pos: str = "a",
       only_hyponyms: bool = False,
       nof_repl_p_cand: int = 2) -> List[str]:
    """
    Data augmentation with `ra`. It takes all the adjectives or nouns in a
    sentence `sent` (candidates) and replaces each of them `nof_repl_p_cand`
    times for a synonym. In particular, if an adjective/noun is selected, then
    the synonym/hyponym will be an adjective/noun.

    parameters:
        - sent (str) sentence to augment.
        - pos (str) POS to replace. Values: 'a' for adjectives, and 's' for
          nouns.
        - only_hyponyms (bool) only replace with hyponyms instead of synonyms.
        - nof_repl_p_cand (int) number of replacements per candidate.
    returns:
        - adv_examples (List[str]) list of augmented sentences.
    """
    adversarial_examples = []

    # Tokenize and pos tag sentence. TODO: the nltk tagger here is different to
    # the one used for other methods such as rsa ir aav (flair tagger) but keeps
    # the same notation. however wordnet uses a different pos notation. Since
    # synonyms and hyponyms are obtained with this last library following the
    # directions of https://hal.archives-ouvertes.fr/hal-02933266, this method
    # might be refactored to use only one POS notation.
    tokens = word_tokenize(sent)
    tags = pos_tag(tokens)

    # 'a' and 's' in wordet is more or less equivalent to 'JJ' and 'NN' in
    # flair.
    if pos == "a" or pos == "s":
        TAG = "JJ"
    else:
        TAG = "NN"

    # Get all adjectives or adverbs and replace them by synonyms or hyponyms.
    words = [(idx, word) for idx, (word, tag) in enumerate(tags) if tag == TAG]
    for idx, word in words:
        candidates = get_synonyms(word, pos=pos, only_hyponyms=only_hyponyms)
        candidates = candidates[:nof_repl_p_cand]
        new_example = tokens.copy()

        for change in candidates:
            new_example[idx] = change
            adversarial_examples.append(" ".join(new_example))
    return adversarial_examples
