import numpy as np
import pytest

from loraplan.probability import distributions as lpd


@pytest.fixture
def getGenerators():
    seed = 11261432
    rng1 = np.random.default_rng(seed=seed)
    rng2 = np.random.default_rng(seed=seed)
    return rng1, rng2


class TestNormal:
    @pytest.mark.parametrize(
        "prop_input, expected",
        [
            ({"loc": 0, "scale": 0, "seed": None}, (0, 0, None)),
            ({"loc": 0, "scale": 0, "seed": 123}, (0, 0, 123)),
            ({"loc": 0, "scale": 0}, (0, 0, None)),
            ({"loc": [0, 0], "scale": 1, "seed": None}, ([0, 0], 1, None)),
            ({"loc": [0, 0], "scale": [1, 1], "seed": None}, ([0, 0], [1, 1], None)),
        ],
    )
    def test_properties(self, prop_input, expected):
        norm = lpd.Normal(**prop_input)
        assert (norm.loc, norm.scale, norm.seed) == expected

    @pytest.mark.parametrize(
        "repr_input, expVals",
        [
            ({"loc": 0, "scale": 0, "seed": None}, (0, 0, None)),
            ({"loc": 0, "scale": 0, "seed": 123}, (0, 0, 123)),
            ({"loc": 0, "scale": 0}, (0, 0, None)),
            ({"loc": [0, 0], "scale": 1, "seed": None}, ([0, 0], 1, None)),
            ({"loc": [0, 0], "scale": [1, 1], "seed": None}, ([0, 0], [1, 1], None)),
        ],
    )
    def test_repr(self, repr_input, expVals):
        """Make a dictionary and compare."""
        norm = lpd.Normal(**repr_input)
        propdict = {"loc": expVals[0], "scale": expVals[1], "seed": expVals[2]}
        expected = "NormalDistribution(%s)" % str(propdict)

        assert norm.__repr__() == expected

    @pytest.mark.parametrize(
        "initKwargs, callKwargs, numpyArgs",
        [
            ({"loc": 0, "scale": 1}, {}, (0, 1)),
            ({"loc": 0, "scale": 1}, {"size": 2}, (0, 1, 2)),
            ({"loc": 0, "scale": 1}, {"size": (5, 5)}, (0, 1, (5, 5))),
            ({"loc": 0, "scale": 1}, {"loc": 5, "scale": 9}, (5, 9)),
            ({"loc": 0, "scale": 1}, {"loc": 5, "scale": 9, "size": 4}, (5, 9, 4)),
        ],
    )
    def test_sample(self, initKwargs, callKwargs, numpyArgs, getGenerators):
        """Get sample via Normal and numpy and compare."""

        rng1, rng2 = getGenerators

        norm = lpd.Normal(**initKwargs)

        ldp_sample = norm.sample(seed=rng1, **callKwargs)
        np_sample = rng2.normal(*numpyArgs)

        assert np.all(ldp_sample == np_sample)
