# causalbundle/empirical/environments.py

from dataclasses import dataclass

@dataclass
class EmpiricalEnvironment:
    """
    Wrapper around a 4.1 EmpiricalCausalModel.

    This does NOT modify empirical estimation logic.
    It only provides an identity for grouping models.
    """

    name: str
    model: object  # EmpiricalCausalModel (from 4.1)
