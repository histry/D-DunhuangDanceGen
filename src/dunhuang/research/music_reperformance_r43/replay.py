"""Exact archived bytes imply exact arrays; preserve the original fallback for other archives."""
from dunhuang.research.complete_r36.core import sha
from dunhuang.research.source_context_r42.contracts import exact_arrays as numerical_exact_arrays

def exact_arrays(first,second):
    if sha(first)==sha(second):return
    numerical_exact_arrays(first,second)
