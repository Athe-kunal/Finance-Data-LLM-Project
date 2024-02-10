import re
import os
import pathlib
from typing import List, Set, Optional
from nltk import sent_tokenize as _sent_tokenize
from nltk import word_tokenize as _word_tokenize
import sys
from functools import lru_cache

if sys.version_info < (3, 8):
    from typing_extensions import Final  # pragma: no cover
else:
    from typing import Final
import nltk
import unicodedata
from unstructured.logger import trace_logger

CACHE_MAX_SIZE: Final[int] = 128
nltk.download("punkt")

ENDS_IN_PUNCT_PATTERN = r"[^\w\s]\Z"
ENDS_IN_PUNCT_RE = re.compile(ENDS_IN_PUNCT_PATTERN)
ENGLISH_WORD_SPLIT_RE = re.compile(r"[\s\-,.!?_\/]+")
NON_LOWERCASE_ALPHA_RE = re.compile(r"[^a-z]")

DIRECTORY = pathlib.Path(__file__).parent.resolve()
# NOTE(robinson) - the list of English words is based on the nlkt.corpus.words corpus
# and the list of English words found here at the link below. Add more words to the text
# file if needed.
# ref: https://github.com/jeremy-rifkin/Wordlist
ENGLISH_WORDS_FILE = os.path.join(DIRECTORY, "english-words.txt")

with open(ENGLISH_WORDS_FILE) as f:
    BASE_ENGLISH_WORDS = f.read().split("\n")

# NOTE(robinson) - add new words that we want to pass for the English check in here
ADDITIONAL_ENGLISH_WORDS: List[str] = []
ENGLISH_WORDS: Set[str] = set(BASE_ENGLISH_WORDS + ADDITIONAL_ENGLISH_WORDS)


def _download_nltk_package_if_not_present(package_name: str, package_category: str):
    """If the required nlt package is not present, download it."""
    try:
        nltk.find(f"{package_category}/{package_name}")
    except LookupError:
        nltk.download(package_name)


@lru_cache(maxsize=CACHE_MAX_SIZE)
def word_tokenize(text: str) -> List[str]:
    """A wrapper around the NLTK word tokenizer with LRU caching enabled."""
    _download_nltk_package_if_not_present(
        package_category="tokenizers", package_name="punkt"
    )
    return _word_tokenize(text)


@lru_cache(maxsize=CACHE_MAX_SIZE)
def sent_tokenize(text: str) -> List[str]:
    """A wrapper around the NLTK sentence tokenizer with LRU caching enabled."""
    _download_nltk_package_if_not_present(
        package_category="tokenizers", package_name="punkt"
    )
    return _sent_tokenize(text)


tbl = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)


def remove_punctuation(s: str) -> str:
    """Removes punctuation from a given string."""
    s = s.translate(tbl)
    return s


def sentence_count(text: str, min_length: Optional[int] = None) -> int:
    """Checks the sentence count for a section of text. Titles should not be more than one
    sentence.

    Parameters
    ----------
    text
        The string of the text to count
    min_length
        The min number of words a section needs to be for it to be considered a sentence.
    """
    sentences = sent_tokenize(text)
    count = 0
    for sentence in sentences:
        sentence = remove_punctuation(sentence)
        words = [word for word in word_tokenize(sentence) if word != "."]
        if min_length and len(words) < min_length:
            trace_logger.detail(  # type: ignore
                f"Sentence does not exceed {min_length} word tokens, it will not count toward "
                "sentence count.\n"
                f"{sentence}",
            )
            continue
        count += 1
    return count


def under_non_alpha_ratio(text: str, threshold: float = 0.5):
    """Checks if the proportion of non-alpha characters in the text snippet exceeds a given
    threshold. This helps prevent text like "-----------BREAK---------" from being tagged
    as a title or narrative text. The ratio does not count spaces.

    Parameters
    ----------
    text
        The input string to test
    threshold
        If the proportion of non-alpha characters exceeds this threshold, the function
        returns False
    """
    if len(text) == 0:
        return False

    alpha_count = len([char for char in text if char.strip() and char.isalpha()])
    total_count = len([char for char in text if char.strip()])
    return ((alpha_count / total_count) < threshold) if total_count > 0 else False


def contains_english_word(text: str) -> bool:
    """Checks to see if the text contains an English word."""
    text = text.lower()
    words = ENGLISH_WORD_SPLIT_RE.split(text)
    for word in words:
        # NOTE(Crag): Remove any non-lowercase alphabetical
        # characters.  These removed chars will usually be trailing or
        # leading characters not already matched in ENGLISH_WORD_SPLIT_RE.
        # The possessive case is also generally ok:
        #   "beggar's" -> "beggars" (still an english word)
        # and of course:
        #   "'beggars'"-> "beggars" (also still an english word)
        word = NON_LOWERCASE_ALPHA_RE.sub("", word)
        if len(word) > 1 and word in ENGLISH_WORDS:
            return True
    return False


def is_possible_title(
    text: str,
    sentence_min_length: int = 5,
    title_max_word_length: int = 12,
    non_alpha_threshold: float = 0.5,
    languages: List[str] = ["eng"],
    language_checks: bool = False,
) -> bool:
    """Checks to see if the text passes all of the checks for a valid title.

    Parameters
    ----------
    text
        The input text to check
    sentence_min_length
        The minimum number of words required to consider a section of text a sentence
    title_max_word_length
        The maximum number of words a title can contain
    non_alpha_threshold
        The minimum number of alpha characters the text needs to be considered a title
    languages
        The list of languages present in the document. Defaults to ["eng"] for English
    language_checks
        If True, conducts checks that are specific to the chosen language. Turn on for more
        accurate partitioning and off for faster processing.
    """
    _language_checks = os.environ.get("UNSTRUCTURED_LANGUAGE_CHECKS")
    if _language_checks is not None:
        language_checks = _language_checks.lower() == "true"

    if len(text) == 0:
        trace_logger.detail("Not a title. Text is empty.")  # type: ignore
        return False

    if text.isupper() and ENDS_IN_PUNCT_RE.search(text) is not None:
        return False

    title_max_word_length = int(
        os.environ.get("UNSTRUCTURED_TITLE_MAX_WORD_LENGTH", title_max_word_length),
    )
    # NOTE(robinson) - splitting on spaces here instead of word tokenizing because it
    # is less expensive and actual tokenization doesn't add much value for the length check
    if len(text.split(" ")) > title_max_word_length:
        return False

    non_alpha_threshold = float(
        os.environ.get("UNSTRUCTURED_TITLE_NON_ALPHA_THRESHOLD", non_alpha_threshold),
    )
    if under_non_alpha_ratio(text, threshold=non_alpha_threshold):
        return False

    # NOTE(robinson) - Prevent flagging salutations like "To My Dearest Friends," as titles
    if text.endswith(","):
        return False

    if "eng" in languages and not contains_english_word(text) and language_checks:
        return False

    if text.isnumeric():
        trace_logger.detail(f"Not a title. Text is all numeric:\n\n{text}")  # type: ignore
        return False

    # NOTE(robinson) - The min length is to capture content such as "ITEM 1A. RISK FACTORS"
    # that sometimes get tokenized as separate sentences due to the period, but are still
    # valid titles
    if sentence_count(text, min_length=sentence_min_length) > 1:
        trace_logger.detail(  # type: ignore
            f"Not a title. Text is longer than {sentence_min_length} sentences:\n\n{text}",
        )
        return False

    return True
