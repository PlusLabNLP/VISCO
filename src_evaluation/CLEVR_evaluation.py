import string

from word2number import w2n


def safe_equal(prediction, answer):
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        print(e)
        return False


def CLEVREvaluate(answer, label, meta_data):
    is_integer = False

    try:
        label = int(label)
        is_integer = True
    except:
        pass

    if is_integer:
        try:
            answer = int(answer)
        except:
            try:
                answer = w2n.word_to_num(''.join([a for a in answer if a not in string.punctuation]))
                answer = str(int(answer))
            except:
                answer = None
        return safe_equal(answer, label)

    else:
        try:
            translator = str.maketrans('', '', string.punctuation)
            answer = answer.translate(translator)

            answer = answer.split(" ")[0]

            answer = answer.lower()
            return safe_equal(answer, label)
        except:
            return False
