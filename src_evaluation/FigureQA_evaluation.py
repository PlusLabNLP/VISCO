from .VSR_evaluation import VSREvaluate


def FigureQAEvaluate(answer, label, meta_data):
    if label == 0:
        label = 'False'
    else:
        assert label == 1
        label = 'True'
    return VSREvaluate(answer, label, meta_data)
