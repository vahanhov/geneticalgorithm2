

from typing import Dict, Any, List, Optional


from dataclasses import dataclass


class DictLikeGetter:
    def __getitem__(self, item):
        return getattr(self, item)





_algorithm_params_slots = {'max_num_iteration','max_iteration_without_improv',
                 'population_size','mutation_probability','elit_ratio','crossover_probability','parents_portion',
    'crossover_type','mutation_type','selection_type'}


@dataclass(init = False)
class AlgorithmParams(DictLikeGetter):

    max_num_iteration: Optional[int] = None
    max_iteration_without_improv: Optional[int] = None

    population_size: int = 100,

    mutation_probability: float = 0.1
    elit_ratio: float = 0.01
    crossover_probability: float = 0.5
    parents_portion: float = 0.3

    crossover_type: str = 'uniform'
    mutation_type: str = 'uniform_by_center'
    selection_type: str = 'roulette'


    @staticmethod
    def from_dict(dct: Dict[str, Any]):

        result = AlgorithmParams()

        for name, value in dct.items():
            if name not in _algorithm_params_slots:
                raise AttributeError(f"name '{name}' does not exists in AlgorithmParams fields")

            setattr(result, name, value)
        return result





