import re


def safe_equal(prediction, answer):
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        print(e)
        return False


def MathVisionEvaluate(answer, label, meta_data):
    if label[0] == "(":
        extracted_label = label[1:-1].split(",")

        try:
            if answer[0] == "(" and answer[-1] == ")":
                extracted_answer = answer[1:-1].split(",")

                extracted_answer = [str(round(float(e))) for e in extracted_answer]

                for i in range(len(extracted_answer)):
                    if safe_equal(extracted_answer[i], extracted_label[i]) == False:
                        return False

                return True
        except:
            return False

    try:
        float(answer)
        is_number = True
    except:
        is_number = False

    if is_number == True:
        if "." in label:
            extracted_answer = str(round(float(answer), 1))
        else:
            extracted_answer = str(round(float(answer)))

        return safe_equal(extracted_answer, label)

    else:
        letter = re.search(r'[a-zA-Z]', answer)
        if letter:
            answer = letter.group()
        else:
            answer = None

        return safe_equal(answer, label)
