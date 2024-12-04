from .VSR_evaluation import VSREvaluate


def PlotQAEvaluate(answer, label, meta_data):
    if label in ['Yes', 'No', ]:
        label = {'Yes': 'True', 'No': 'False'}[label]
        return VSREvaluate(answer, label, meta_data)
    else:
        try:
            label = float(label)
            answer = float(answer)
            return label == answer
        except:
            return str(label).lower() == str(answer).lower()
