import numpy as np
import pytest

from loraplan.probability import distributions as lpd


@pytest.fixture
def getGenerators():
    seed = 11261432
    rng1 = np.random.default_rng(seed=seed)
    rng2 = np.random.default_rng(seed=seed)
    return rng1, rng2


# --------------------------------------
# Test Normal
# --------------------------------------


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


# --------------------------------------
# Test Choice
# --------------------------------------


class TestChoice:
    @pytest.mark.parametrize(
        "prop_input, expected",
        [
            ({"a": 0, "p": 1, "seed": None}, (0, 1, True, None)),
            ({"a": [0, 1], "p": None, "seed": None}, ([0, 1], None, True, None)),
            (
                {"a": [0, 2], "p": [0.4, 0.6], "replace": False},
                ([0, 2], [0.4, 0.6], False, None),
            ),
            ({"a": [0, 1], "p": None, "seed": 123}, ([0, 1], None, True, 123)),
            ({"a": None, "p": [1, 2], "seed": 123}, (None, [1, 2], True, 123)),
        ],
    )
    def test_properties(self, prop_input, expected):
        choice = lpd.Choice(**prop_input)
        assert (choice.a, choice.p, choice.replace, choice.seed) == expected

    @pytest.mark.parametrize(
        "repr_input, expVals",
        [
            ({"a": 0, "p": 0, "seed": None}, (0, 0, True, None)),
            ({"a": 0, "p": 0, "seed": 123}, (0, 0, True, 123)),
            ({"a": 0, "p": 0}, (0, 0, True, None)),
            ({"a": [0, 0], "p": 1, "replace": False}, ([0, 0], 1, False, None)),
            ({"a": [0, 0], "p": [1, 1], "seed": None}, ([0, 0], [1, 1], True, None)),
        ],
    )
    def test_repr(self, repr_input, expVals):
        """Make a dictionary and compare."""
        choice = lpd.Choice(**repr_input)
        propdict = {
            "a": expVals[0],
            "p": expVals[1],
            "replace": expVals[2],
            "seed": expVals[3],
        }
        expected = "DiscreteDistribution(%s)" % str(propdict)

        assert choice.__repr__() == expected

    @pytest.mark.parametrize(
        "initKwargs, callKwargs, numpyKwargs",
        [
            ({"a": [0, 1]}, {}, {"a": [0, 1]}),
            ({"a": [0, 1], "p": [0.1, 0.9]}, {}, {"a": [0, 1], "p": [0.1, 0.9]}),
            ({"a": [0, 1]}, {"p": [0.1, 0.9]}, {"a": [0, 1], "p": [0.1, 0.9]}),
            ({"a": [0, 1]}, {"a": [3, 2, 4]}, {"a": [3, 2, 4]}),
            (
                {"a": [0, 1]},
                {"replace": False, "size": 2},
                {"a": [0, 1], "replace": False, "size": 2},
            ),
        ],
    )
    def test_sample(self, initKwargs, callKwargs, numpyKwargs, getGenerators):

        rng1, rng2 = getGenerators

        disc = lpd.Choice(**initKwargs)

        lpd_sample = disc.sample(**callKwargs, seed=rng1)
        np_sample = rng2.choice(**numpyKwargs)

        assert np.all(lpd_sample == np_sample)


# --------------------------------------
# Test Uniform
# --------------------------------------


class TestUniform:
    @pytest.mark.parametrize(
        "prop_input, expected",
        [
            ({"low": 0, "high": 1, "seed": None}, (0, 1, None)),
            ({"low": 0, "high": 1, "seed": 123}, (0, 1, 123)),
            ({"low": [1, 1], "high": [2, 5]}, ([1, 1], [2, 5], None)),
        ],
    )
    def test_properties(self, prop_input, expected):
        unif = lpd.Uniform(**prop_input)
        assert (unif.low, unif.high, unif.seed) == expected

    @pytest.mark.parametrize(
        "prop_input, expected",
        [
            ({"low": 0, "high": 1, "seed": None}, (0, 1, None)),
            ({"low": 0, "high": 1, "seed": 123}, (0, 1, 123)),
            ({"low": [1, 1], "high": [2, 5]}, ([1, 1], [2, 5], None)),
        ],
    )
    def test_repr(self, prop_input, expected):
        unif = lpd.Uniform(**prop_input)

        data = {"low": expected[0], "high": expected[1], "seed": expected[2]}
        text = f"UniformDistribution({data})"

        assert unif.__repr__() == text

    @pytest.mark.parametrize(
        "initKwargs, callKwargs, numpyKwargs",
        [
            ({"low": 0, "high": 1}, {}, {"low": 0, "high": 1}),
            ({"low": 0, "high": 1}, {"low": 5, "high": 10}, {"low": 5, "high": 10}),
            ({"low": 0, "high": 1}, {"size": 5}, {"low": 0, "high": 1, "size": 5}),
            (
                {"low": 0, "high": 1},
                {"size": [5, 3]},
                {"low": 0, "high": 1, "size": [5, 3]},
            ),
        ],
    )
    def test_sample(self, initKwargs, callKwargs, numpyKwargs, getGenerators):

        rng1, rng2 = getGenerators

        unif = lpd.Uniform(**initKwargs)

        lpd_sample = unif.sample(**callKwargs, seed=rng1)
        np_sample = rng2.uniform(**numpyKwargs)

        assert np.all(lpd_sample == np_sample)
