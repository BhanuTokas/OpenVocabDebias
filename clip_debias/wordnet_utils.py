"""
wordnet_utils.py — WordNet helpers for automated concept word extraction.
"""

from __future__ import annotations

import random
from typing import List, Set

from nltk.corpus import wordnet as wn


def get_synonyms(word: str, filter_by_name: bool = True) -> List[str]:
    """
    Return all unique lemma names across every synset of 'word'.

    Parameters
    ----------
    word           : the input word (e.g. "water")
    filter_by_name : if True (default), only include synsets whose synset name
                     contains the query word.  This excludes synsets that merely
                     list the word as a secondary lemma but are really about
                     something else (e.g. "urine.n.01" lists "water" as a lemma
                     but should not be included when querying "water").

    Returns
    -------
    List of unique, space-separated lemma strings (underscores replaced).
    """
    seen: Set[str] = set()
    results: List[str] = []

    for synset in wn.synsets(word):
        if filter_by_name and word not in synset.name():
            continue
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if name not in seen:
                seen.add(name)
                results.append(name)

    return results


def get_random_base_nouns(
    n: int,
    exclude_words: Set[str],
    seed: int = 42,
) -> List[str]:
    """
    Sample n random noun lemmas from WordNet, excluding any word in exclude_words.

    Parameters
    ----------
    n             : number of words to return
    exclude_words : set of lowercase words to skip (exact match after underscore→space)
    seed          : random seed for reproducibility

    Returns
    -------
    List of n unique, space-separated noun lemma strings.
    """
    exclude_norm = {w.replace("_", " ").lower() for w in exclude_words}

    all_synsets = list(wn.all_synsets(pos=wn.NOUN))
    rng = random.Random(seed)
    rng.shuffle(all_synsets)

    results: List[str] = []
    seen: Set[str] = set()

    for synset in all_synsets:
        if len(results) >= n:
            break
        name = synset.lemmas()[0].name().replace("_", " ").lower()
        # Skip if it overlaps with any excluded word
        if any(exc in name or name in exc for exc in exclude_norm):
            continue
        if name not in seen:
            seen.add(name)
            results.append(name)

    return results
