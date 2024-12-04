import re


def safe_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types
    """
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        print(e)
        return False


def WeMathEvaluate(answer, label, meta_data):
    extraction = None

    letter = re.findall(r'\(([a-zA-Z])\)', answer)
    if len(letter) > 0:
        extraction = letter[0].upper()

    if extraction is None:
        letter = re.search(r'\"[a-zA-Z]\"', answer)
        if letter:
            extraction = letter.group()

    if extraction is None:  # we don't have options, we can't match options, so we just extract first letter anyway
        letter = re.search(r'[a-zA-Z]', answer)
        if letter:
            extraction = letter.group()

    return safe_equal(extraction, label)
