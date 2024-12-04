def VSREvaluate(text, label, meta_data):
    # Only keep the first sentence
    if text.find('.') != -1:
        text = text.split('.')[0]

    text = text.replace(',', '')
    words = text.split(' ')
    if 'False' in words or 'false' in words or 'No' in words or 'not' in words or 'no' in words:
        answer = 'False'
    else:
        answer = 'True'

    if answer == label:
        return True
    else:
        return False
