"""Reserved 30-second physical multi-device Mandelbulb scaling showcase.

Upload correctness is checked by test_image_workloads.py and the real-browser
script demos/mandelbulb_upload_check.py. That route does not yet implement the
requested automatic cost probe or measure scaling across physical GPUs.
"""

import pytest


@pytest.mark.slow
@pytest.mark.parametrize("node_count", [2, 3, 4])
def test_attached_mandelbulb_showcase(node_count):
    pytest.skip(
        "30-second cost-probed 2/3/4 physical-GPU scaling showcase is not implemented; upload correctness is covered separately"
    )
