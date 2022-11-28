import pytest

from loraplan.interference import IndependentLoRaGenerator, LoRaParameters, Traffic
from loraplan.probability import point_processes


@pytest.fixture
def dummyGen():
    """Make a dummy IndependentLoRaGenerator"""
    params = LoRaParameters()

    return IndependentLoRaGenerator.from_parameters(params)


@pytest.mark.parametrize(
    "timeWinKwargs, startTimes, expected",
    [
        ({"tMin": 1, "tMax": 2, "buffer": 1}, [0, 1.5, 2.5], [False, True, False]),
        ({"tMin": 5, "tMax": 10, "buffer": 1}, [7], [True]),
        ({"tMin": 5, "tMax": 10, "buffer": 1}, [0], [False]),
        (
            {"tMin": 5, "tMax": 10, "buffer": 5},
            [4, 8, 12, 16],
            [False, True, False, False],
        ),
        ({"tMin": 5, "tMax": 10, "buffer": 1}, [], []),
    ],
)
class TestIndependentLoRaGenerator:
    def test_notBuffer(self, timeWinKwargs, startTimes, expected, dummyGen):

        gen = dummyGen
        gen.arrivals.timeWindow = point_processes.TimeWindow(**timeWinKwargs)

        traffic = Traffic(
            len(startTimes),
            start=startTimes,
            airtime=startTimes,
            channel=startTimes,
            sf=startTimes,
            power=startTimes,
        )

        assert all(gen.notBuffer(traffic) == expected)
