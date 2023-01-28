from typing import List
from ISHate.augmentation.utils import get_synonyms, stop_words
import random
import re


def synonym_replacement(words: List[str], n: int) -> List[str]:
    """
    Replace `min(n, len(words)` non-stopwords words in the sentence composed by
    a list of `words` with synonyms from wordnet.

    parameters:
        - words (List[str]) list of words representing a sentence.
        - n (int) number of words to replace.

    returns:
        new_words (List[str]) list of words in the same order as `words` but
        with the corresponding n replacements performed randomly.

    """
    new_words = words.copy()

    # Remove stopwords and duplicates.
    random_word_list = list(
        set([word for word in words if word not in stop_words]))

    # Shuffle word list
    random.shuffle(random_word_list)
    num_replaced = 0

    # Replace up to `n` words of `new_words` for a random synonym.
    for random_word in random_word_list:
        synonyms = get_synonyms(word=random_word,
                                pos=None,
                                only_hyponyms=False)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [
                synonym if word == random_word else word for word in new_words
            ]
            num_replaced += 1
        if num_replaced >= n:  #only replace up to n words
            break

    # TODO: Next comment was written by EDA authors. Check why we need these two lines

    # this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_only_chars(line: str) -> str:
    """
    Receives an string `line` and removes any non utf-8 character and leftovers
    spaces.

    parameters:
        - line (str) string to clean

    returns:
        - clean_line (str)
    """

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")

    # Replace hyphens, tabs, and line breaks with spaces.
    line = line.replace("-", " ")
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    # Check if there are only utf-8 chars.
    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    # Delete extra spaces
    clean_line = re.sub(' +', ' ', clean_line)
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


def random_deletion(words: List[str], p: float) -> List[str]:
    """
    Randomly deletes words from the sentence with probability p.

    If `words` has only one word, it keeps it unchanged. If all words are
    deleted it returns a random word from `words`.

    parameters:
        - `words` (List[str]) list of words.
        - `p` (float) probability, that is, a value between 0 and 1.

    returns:
        - `new_words` (List[str]) list of words in the same order as `words` but
        with the corresponding deletions performed randomly.
    """

    # If there's only one word, don't delete it
    if len(words) == 1:
        return words

    # Randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # If you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


def random_swap(words: List[str], n: int) -> List[str]:
    """
    Randomly swap two words in the sentence n times

    parameters:
        - `words` (List[str]) list of words.
        - `n` (int) number of times to swap.

    returns:
        new_words (List[str]) a copy of `words` with `n` swaps.
    """
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words: List[str]) -> List[str]:
    """
    Randomly choose two positions i and j in `new_words` and swap them. It might
    return a list with no changes if i and j are the same two times
    consecutively.

    parameters:
        - `new_words` (List[str]) list of words.

    returns:
        - `new_words` (List[str]) a copy of the input with two words swapped.
    """
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0

    # Loop to choose at least two distinct random words.
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words

    # Make swap, with multiple assignment.
    new_words[random_idx_1], new_words[random_idx_2] = new_words[
        random_idx_2], new_words[random_idx_1]
    return new_words


def random_insertion(words: List[str], n: int) -> List[str]:
    """
    Randomly inserts n words into the sentence.

    parameters:
        - `words` (List[str]) list of words.

    returns:
        - `new_words` (List[str]) a copy of the input with `n` insertions.
    """
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words: List[str]) -> None:
    """
    Randomly takes a word in `new_words`, finds its synonym, and inserts the
    synonym in a random position.

    parameters:
        - `new_words` (List[str]) list of words.
    """
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word, pos=None, only_hyponyms=False)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


def eda(sent: str,
        alpha_sr: int = 0.1,
        alpha_ri: int = 0.1,
        alpha_rs: int = 0.1,
        p_rd: int = 0.1,
        num_aug: int = 2,
        add_original: bool = False):
    """
    Data augmentation with method `eda`. Given an input sentence `sent`, i) it
    randomly replaces a non-stopword expression with a synonym using Word-net;
    ii) inserts a synonym of a non-stopword word in a random position; iii)
    chooses two words of the sentence and swap their positions; iv) removes each
    word in the sentence with a certain probability. Only one of the four
    operations at a time is applied to a sentence.

    parameters:
        - `sent` (str) sentence to augment.
        - `alpha_sr` (int) percent of the words in a sentence to be changed in
          synonym replacement.
        - `alpha_ri` (int) percent of the words in a sentence to be changed in
          random insertion.
        - `alpha_rs` (int) percent of the words in a sentence to be changed in
          random swap.
        - `p_rd` (int) percent of the words in a sentence to be changed in
          random deletion.
        - `num_aug` (int) number of augmented sentences per technique.
        - `add_original` (bool) add original sentence to the output.

    returns:
        - augmented_sentences (List[str]) list of augmented sentences.
    """

    only_char_sentence = get_only_chars(sent)
    words = only_char_sentence.split(' ')
    words = [word for word in words if word != '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1

    # synonym replacement
    if alpha_sr > 0:
        n_sr = max(1, int(alpha_sr * num_words))
        for _ in range(num_new_per_technique):
            a_words = synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words))

    # random insertion
    if alpha_ri > 0:
        n_ri = max(1, int(alpha_ri * num_words))
        for _ in range(num_new_per_technique):
            a_words = random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words))

    # random swap
    if alpha_rs > 0:
        n_rs = max(1, int(alpha_rs * num_words))
        for _ in range(num_new_per_technique):
            a_words = random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words))

    # random deletion
    if p_rd > 0:
        for _ in range(num_new_per_technique):
            a_words = random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [
        get_only_chars(sentence) for sentence in augmented_sentences
    ]
    random.shuffle(augmented_sentences)

    # Trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [
            s for s in augmented_sentences if random.uniform(0, 1) < keep_prob
        ]

    if add_original:
        #append the original sentence
        augmented_sentences.append(sent)

    return augmented_sentences