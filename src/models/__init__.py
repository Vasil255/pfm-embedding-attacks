from .abmil import ABMIL
from .transmil import TransMIL

AGGREGATORS = {"ABMIL": ABMIL, "TransMIL": TransMIL}


def build_model(name, n_classes, L):
    return AGGREGATORS[name](n_classes=n_classes, L=L)
