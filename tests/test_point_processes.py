import numpy as np
import pytest

from loraplan.probability import point_processes as lpp


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
