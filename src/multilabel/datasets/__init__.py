"""Multi-label datasets for chest X-ray classification."""

from .chexpert_multilabel import CheXpertMultiLabel
from .cxr8_multilabel import CXR8MultiLabel

__all__ = ['CheXpertMultiLabel', 'CXR8MultiLabel']
