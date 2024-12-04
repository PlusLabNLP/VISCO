import re


def safe_equal(prediction, answer):
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        print(e)
        return False


def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))


def convert_string(input_string):
    # Define the regular expression pattern
    pattern = r'[^0-9.,/]'

    # Use re.sub() to replace unwanted characters
    result = re.sub(pattern, '', input_string)

    return result


def MMVetEvaluate(answer, label, meta_data):
    labels = label.split("<OR>")

    for label in labels:
        if has_numbers(label):
            label = convert_string(label)
            if "," in label or "/" in label:
                continue

            try:
                if "." in label:
                    precision = len(label.split(".")[1])
                    answer = str(round(float(convert_string(str(answer))), precision))
                else:
                    answer = str(round(float(convert_string(str(answer)))))
            except:
                continue

            if safe_equal(answer, label):
                return True

        else:
            if safe_equal(answer, label):
                return True

    return False