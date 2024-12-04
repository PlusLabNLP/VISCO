import re

from .MathVista_evaluation import get_most_similar


def safe_equal(prediction, answer):
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        print(e)
        return False


def ScienceQA_evaluate(extraction, label, meta_data):
    choices = meta_data['choices']

    # extract "A" from "(A) text"
    letter = re.findall(r'\(([a-zA-Z])\)', extraction)
    if len(letter) > 0:
        extraction = letter[0].upper()

    # also try to extract \"A\" from '"A"'
    letter = re.search(r'\"[a-zA-Z]\"', extraction)
    if letter:
        extraction = letter.group()

    options = [chr(ord('A') + i) for i in range(len(choices))]
    assert label in options

    if extraction not in options:
        # select the most similar option
        choice = get_most_similar(extraction, choices)
        extraction = options[choices.index(choice)]
    assert extraction in options

    return safe_equal(extraction, label)
