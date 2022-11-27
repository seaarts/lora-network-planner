import numpy as np
import pytest

from loraplan.probability import point_processes as lpp


@pytest.fixture
def getGenerators():
    """NumPy RNGs for use when testing randomness."""
    seed = 11261432
    rng1 = np.random.default_rng(seed=seed)
    rng2 = np.random.default_rng(seed=seed)
    return rng1, rng2


class TestRectangle:
    @pytest.mark.parametrize(
        "initKwargs, expected",
        [
            ({"bbox": [0, 0, 1, 1]}, ((0, 0, 1, 1), None)),
            ({"bbox": [0, 0, 1, 1], "name": "Bob"}, ((0, 0, 1, 1), "Bob")),
        ],
    )
    def test_attrs(self, initKwargs, expected):
        rect = lpp.Rectangle(**initKwargs)
        assert (rect.bbox, rect.name) == expected

    @pytest.mark.parametrize(
        "initKwargs, expected",
        [
            ({"bbox": [0, 0, 1, 1]}, (1, 1, 1, 1)),
            ({"bbox": [0, 0, 2, 2]}, (2, 2, 4, 4)),
        ],
    )
    def test_props(self, initKwargs, expected):
        rect = lpp.Rectangle(**initKwargs)
        assert (rect.width, rect.height, rect.area, rect.measure) == expected

    @pytest.mark.parametrize(
        "initKwargs, callKwargs, expected",
        [
            ({"bbox": [0, 0, 1, 1]}, {"nPoints": 1}, (1, 2)),
            ({"bbox": [0, 0, 1, 1]}, {"nPoints": 3}, (3, 2)),
            ({"bbox": [0, 0, 1, 1]}, {"nPoints": 11}, (11, 2)),
        ],
    )
    def test_uniform(self, initKwargs, callKwargs, expected):
        """Just check shapes."""

        rect = lpp.Rectangle(**initKwargs)

        my_sample = rect.uniform(**callKwargs, seed=None)
        assert my_sample.shape == expected

    @pytest.mark.parametrize(
        "initKwargs, callKwargs",
        [
            ({"bbox": [0, 0, 1, 1]}, {"intensity": 1}),
            ({"bbox": [0, 0, 2, 2]}, {"intensity": 1}),
            ({"bbox": [0, 1, 99, 12]}, {"intensity": 1}),
        ],
    )
    def test_poisson(self, initKwargs, callKwargs, getGenerators):
        """Compare shape to (Poisson(measure), 2)

        This should be sufficient given ``Rectangle.area`` presumably works at this stage.
        """
        rng1, rng2 = getGenerators

        rect = lpp.Rectangle(**initKwargs)

        my_sample = rect.poisson(**callKwargs, seedNum=rng1)
        nPoints = np.atleast_1d(my_sample).shape[0]  # nr points
        np_sample = rng2.poisson(lam=rect.area)

        assert nPoints == np_sample
