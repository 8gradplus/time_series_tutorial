from itertools import product


class Compose:
    """ Composition of functions """

    def __init__(self, *fs):
        self.fs = fs

    def __call__(self, x):
        result = x
        for f in self.fs:
            result = f(result)
        return result


class Split:
    """Split of functions 1:n"""

    def __init__(self, *fs):
        self.fs = fs

    def __call__(self, x):
        return [f(x) for f in self.fs]


class PipelineGraph:
    """Compute cartesian product (tree) of functions"""

    def __init__(self, *fs):
        self.fs = fs

    def __call__(self, x):
        cartesian_product = product(*self.fs)
        return self._compute_paths(cartesian_product, x)

    @staticmethod
    def _compute_paths(iterable, x):
        results = {}
        for functions in iterable:
            result = Composition(*functions)(x)
            results[functions] = result
        return results

