from typing import List
from flair.data import Sentence
from ISHate.augmentation.utils import tagger_pos, scalarity_list, replace_exp_to_sent
import random


def rsa(sent: str, nof_repl_p_cand: int = 2) -> List[str]:
    """
    Data augmentation with method `rsa`. It selects all the adverbs of pos type
    'RB' (candidates) in `sent`. For each candidate, it generates
    `nof_repl_p_cand` by replacing the candidate with an scalar adverb.

    parameters:
        - sent (str) sentence to augment.
        - nof_repl_p_cand (int) number of replacements per candidate.
    returns:
        - adv_examples (List[str]) list of augmented sentences.
    """
    # Create a flair instance Sentence.
    sent_ = Sentence(sent)

    # Tag sentence with POS.
    tagger_pos.predict(sent_)

    # Filter all adverbs in `sent`
    advs_in_sent = [
        label for label in sent_.get_labels("pos") if label.value == "RB"
    ]

    # Randomly choose `nof_repl_p_cand` adverbs from `scalarity_list` and
    # replace them for each candidate.
    adv_examples = []
    for adv in advs_in_sent:
        # NOTE: scalarity_list contains adverbs that can go after or before an
        # adj/verb. In this case, we combine them in just one list.
        candidates = random.sample(scalarity_list["BEFORE"] +
                                   scalarity_list["AFTER"],
                                   k=nof_repl_p_cand)

        examples = [
            replace_exp_to_sent(sent=sent,
                                exp=change,
                                start_pos=adv.data_point.start_position,
                                end_pos=adv.data_point.end_position)
            for change in candidates
        ]
        adv_examples.extend(examples)
    return adv_examples