import random
from nltk import word_tokenize
from typing import List
from ISHate.augmentation.utils import special_tokens


def ri(sent: str, nof_repl_p_cand: int = 2) -> List[str]:
    """
    Data augmentation with method `ri`. It replaces a list of manually-crafted
    expressions often used in HS messages (in-domain exp. not captured by `rne`)
    with other semantically similar expressions. That is, it checks all the
    ocurrences of in-domain expressions in `sent` (candidates), and generates
    `nof_repl_p_cand` sentences per each candidate by changing it for another
    manually-collected expression.

    parameters:
        - `sent` (str) sentence to augment.
        - `nof_repl_p_cand` (int) number of replacements per candidate.
    returns:
        adv_examples (List[str]) list of augmented examples.
    """
    adversarial_examples = []

    # Lower and tokenize sentence.
    sent = sent.lower()
    tokens = word_tokenize(sent)

    # Filter out ocurrences of in-domain expressions.
    words = [(idx, token) for idx, token in enumerate(tokens)
             if token in special_tokens.keys()]
    for idx, word in words:
        candidates = random.sample(special_tokens[word],
                                   k=min(len(special_tokens[word]),
                                         nof_repl_p_cand))
        for change in candidates:
            new_example = tokens.copy()
            new_example[idx] = change
            adversarial_examples.append(" ".join(new_example))

    return adversarial_examples
