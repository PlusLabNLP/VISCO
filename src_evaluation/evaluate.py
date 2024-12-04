import os

from src_evaluation.CLEVR_evaluation import CLEVREvaluate
from src_evaluation.EmbSpatial_evaluation import EmbSpatial_evaluation
from src_evaluation.FigureQA_evaluation import FigureQAEvaluate
from src_evaluation.GQA_evaluation import GQAEvaluate
from src_evaluation.HallusionBench_evaluation import HallusionBenchEvaluate
from src_evaluation.MMMU_evaluation import MMMU_evaluate
from src_evaluation.MMVet_evaluation import MMVetEvaluate
from src_evaluation.MathVision_Evaluation import MathVisionEvaluate
from src_evaluation.MathVista_evaluation import MathVistaEvaluate
from src_evaluation.POPE_evaluation import POPEEvaluate
from src_evaluation.PlotQA_evaluation import PlotQAEvaluate
from src_evaluation.SceMQA_evaluation import SceMQA_evaluate
from src_evaluation.ScienceQA_evaluation import ScienceQA_evaluate
from src_evaluation.TallyQA_evaluation import TallyQAEvaluate
from src_evaluation.VQA_evaluation import VQAEvaluate
from src_evaluation.VSR_evaluation import VSREvaluate
from src_evaluation.WeMathEvaluation import WeMathEvaluate


def get_evaluation_method(dataset):
    if dataset == "MathVista":
        return MathVistaEvaluate
    elif dataset == "POPE":
        return POPEEvaluate
    elif dataset == "HallusionBench":
        return HallusionBenchEvaluate
    elif dataset == "WeMath":
        return WeMathEvaluate
    elif dataset == "MathVision":
        return MathVisionEvaluate
    elif dataset == "MMVet":
        return MMVetEvaluate
    elif dataset == "MMMU":
        return MMMU_evaluate
    elif dataset == "ScienceQA":
        return ScienceQA_evaluate
    elif dataset == "SceMQA":
        return SceMQA_evaluate
    elif dataset == "EmbSpatial":
        return EmbSpatial_evaluation
    elif dataset == "TallyQA":
        return TallyQAEvaluate
    elif dataset == "VSR":
        return VSREvaluate
    elif dataset == "TextVQA":
        return VQAEvaluate
    elif dataset == "DocVQA":
        return VQAEvaluate
    elif dataset == "GQA":
        return GQAEvaluate
    elif dataset == "CLEVR":
        return CLEVREvaluate
    elif dataset == "ChartQA":
        return VQAEvaluate
    elif dataset == "FigureQA":
        return FigureQAEvaluate
    elif dataset == "PlotQA":
        return PlotQAEvaluate
    else:
        raise ValueError(f"Dataset not found: {dataset}")


def evaluate(pred, label, meta_data):
    evaluate_func = get_evaluation_method(meta_data['src_dataset'])
    return evaluate_func(pred, label, meta_data)