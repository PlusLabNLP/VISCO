import re

from Levenshtein import distance


def get_most_similar(prediction, choices):
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]
    # return min(choices, key=lambda choice: distance(prediction, choice))


def normalize_extracted_answer(extraction, choices, question_type, answer_type, precision):
    """
    Normalize the extracted answer to match the answer type
    """
    if question_type == 'multi_choice':
        # make sure the extraction is a string
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except:
                extraction = ""

        # extract "A" from "(A) text"
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        # also try to extract \"A\" from '"A"'
        letter = re.search(r'\"[a-zA-Z]\"', extraction)
        if letter:
            extraction = letter.group()

        options = [chr(ord('A') + i) for i in range(len(choices))]

        if extraction in options:
            # convert option letter to text, e.g. "A" -> "text"
            ind = options.index(extraction)
            extraction = choices[ind]
        else:
            # select the most similar option
            extraction = get_most_similar(extraction, choices)
        assert extraction in choices

    elif answer_type == 'integer':
        try:
            extraction = str(int(float(extraction)))
        except:
            extraction = None

    elif answer_type == 'float':
        try:
            extraction = str(round(float(extraction), int(precision)))
        except:
            extraction = None

    elif answer_type == 'list':
        try:
            extraction = str(extraction)
        except:
            extraction = None

    return extraction


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


def MathVistaEvaluate(pred, label, meta_data):
    normalized_answer = normalize_extracted_answer(
        pred, meta_data["choices"], meta_data["question_type"], meta_data["answer_type"], meta_data["precision"]
    )
    correct = safe_equal(normalized_answer, label)
    return correct
