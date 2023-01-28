from flair.data import Sentence
from flair.embeddings import DocumentPoolEmbeddings
from typing import List, Dict
from scipy import spatial
from ISHate.augmentation.utils import tagger_ner, fasttext_doc_emb, vectors_list, replace_exp_to_sent,
import torch


def get_word_vector(word: str,
                    doc_emb: DocumentPoolEmbeddings) -> torch.Tensor:
    """
    Given a word and a flair embedding returns a vector that represents it.
    """
    word = Sentence(word, use_tokenizer=True)
    doc_emb.embed(word)
    return word.embedding.to("cpu")


def find_closest_embeddings(vector_list: Dict[str, torch.Tensor],
                            v: torch.Tensor) -> List[str]:
    """
    Given a vector `v` and a dictionary `vector_list` where keys are words and
    the values are vectors, return the `vector_list` keys (words) in descending
    order.

    One word w1 in `vector_list.keys()` is lower than w2 in `vector_list.keys()`
    if and only if the representation vector of w1, vector_list[w1], is closer
    in euclidean distance to `v` than the representation of w2 is to `v`.
    """
    return sorted(
        vector_list.keys(),
        key=lambda word: spatial.distance.euclidean(vector_list[word], v),
    )


def rne(sent: str,
        cand_ner_name: str = "MISC",
        nof_repl_p_cand: int = 2) -> List[str]:
    """
    Data augmentation with method `rne`. It replaces a named entity (PER, LOC,
    ORG, and MISC) in the input sentence `sent`. A candidate NE in a sentence is
    replaced by another one according to a previously collected list of NEs.
    Then, the most similar NE is selected by using pre-trained FastText
    embeddings. Per each candidate it creates `nof_repl_p_cand` sentences. For
    example, if `ner_name = 'PER'`, `nof_repl_p_cand = 5` and `sent` has 2 named
    entities PER, `rne` generates 2 * 5 sentences.

    parameters:
        - `sent` (str) sentence to augment
        - `ner_name` (str) named entity. values `PER`, `LOC`, `ORG`, and `MISC`.
        - `nof_repl_p_cand` (int) number of replacements for named entity of
          `ner_name` type found.
    returns:
        - adv_examples (List[str]) list of augmented sentences.
    """
    # Create a flair instance Sentence.
    sent_ = Sentence(sent)

    # Tag sentence with NEs.
    tagger_ner.predict(sent_)

    adv_examples = []
    for label in sent_.get_labels("ner"):
        ner_type = label.value

        # Check if a tagged entity word is of type `ner_name`, that is, a
        # possible candidate.
        if ner_type == cand_ner_name:

            # Get the start and end positions of the entity.
            ner_text = label.data_point.text
            ner_start = label.data_point.start_position
            ner_end = label.data_point.end_position

            # Find FastText word embedding of the entity.
            w_vector = get_word_vector(ner_text, doc_emb=fasttext_doc_emb)

            # Get the most similar words to the entity in `vector_list`.
            candidates = find_closest_embeddings(vectors_list[ner_type],
                                                 w_vector)

            # Avoid replacement of the entity for itself.
            if ner_text in candidates:
                candidates.remove(ner_text)

            # Make only `nof_repl_p_cand` replacements.
            candidates = candidates[:nof_repl_p_cand]

            # Create `nof_repl_p_cand` sentences by replacing the entity for the
            # most similar words found in the `vector_list`.
            examples = [
                replace_exp_to_sent(sent=sent,
                                    exp=change,
                                    start_pos=ner_start,
                                    end_pos=ner_end) for change in candidates
            ]

            # Collect all augmented sentences.
            adv_examples.extend(examples)
    return adv_examples