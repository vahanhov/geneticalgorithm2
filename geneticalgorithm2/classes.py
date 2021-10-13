

from typing import Dict, Any, List, Optional, Union, Callable, Tuple


from dataclasses import dataclass
import warnings

import numpy as np

from .crossovers import Crossover
from .mutations import Mutations
from .selections import Selection

from .utils import can_be_prob


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

    crossover_type: Union[str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = 'uniform'
    mutation_type: Union[str, Callable[[float, float, float], float]] = 'uniform_by_center'
    selection_type: Union[str, Callable[[np.ndarray, int], np.ndarray]] = 'roulette'



    def _check_if_valid(self):

        assert int(self.population_size) > 0, f"population size must be integer and >0, not {self.population_size}"
        assert (can_be_prob(self.parents_portion)), "parents_portion must be in range [0,1]"
        assert (can_be_prob(self.mutation_probability)), "mutation_probability must be in range [0,1]"
        assert (can_be_prob(self.crossover_probability)), "mutation_probability must be in range [0,1]"
        assert (can_be_prob(self.elit_ratio)), "elit_ratio must be in range [0,1]"

        if self.max_iteration_without_improv is not None and self.max_iteration_without_improv < 1:
            warnings.warn(f"max_iteration_without_improv is {self.max_iteration_without_improv} but must be None or int > 0")
            self.max_iteration_without_improv = None





    def get_CMS(self):

        crossover = self.crossover_type
        mutation = self.mutation_type
        selection = self.selection_type

        C, M, S = None, None, None

        if type(crossover) == str:
            if crossover == 'one_point':
                C = Crossover.one_point()
            elif crossover == 'two_point':
                C = Crossover.two_point()
            elif crossover == 'uniform':
                C = Crossover.uniform()
            elif crossover == 'segment':
                C = Crossover.segment()
            elif crossover == 'shuffle':
                C = Crossover.shuffle()
            else:
                raise Exception(f"unknown type of crossover: {crossover}")
        else:
            C = crossover

        if type(mutation) == str:
            if mutation == 'uniform_by_x':
                M = Mutations.uniform_by_x()
            elif mutation == 'uniform_by_center':
                M = Mutations.uniform_by_center()
            elif mutation == 'gauss_by_center':
                M = Mutations.gauss_by_center()
            elif mutation == 'gauss_by_x':
                M = Mutations.gauss_by_x()
            else:
                raise Exception(f"unknown type of mutation: {mutation}")
        else:
            M = mutation

        if type(selection) == str:
            if selection == 'fully_random':
                S = Selection.fully_random()
            elif selection == 'roulette':
                S = Selection.roulette()
            elif selection == 'stochastic':
                S = Selection.stochastic()
            elif selection == 'sigma_scaling':
                S = Selection.sigma_scaling()
            elif selection == 'ranking':
                S = Selection.ranking()
            elif selection == 'linear_ranking':
                S = Selection.linear_ranking()
            elif selection == 'tournament':
                S = Selection.tournament()
            else:
                raise Exception(f"unknown type of selection: {selection}")
        else:
            S = selection

        return C, M, S





    @staticmethod
    def from_dict(dct: Dict[str, Any]):

        result = AlgorithmParams()

        for name, value in dct.items():
            if name not in _algorithm_params_slots:
                raise AttributeError(f"name '{name}' does not exists in AlgorithmParams fields")

            setattr(result, name, value)
        return result







