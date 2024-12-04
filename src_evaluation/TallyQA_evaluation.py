import string

from word2number import w2n


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


def TallyQAEvaluate(answer, label, meta_data):
    try:
        answer = str(int(float(answer)))
    except:
        try:
            answer = w2n.word_to_num(''.join([a for a in answer if a not in string.punctuation]))
            answer = str(int(answer))
        except:
            answer = None

    label = str(int(float(label)))
    return safe_equal(answer, label)
