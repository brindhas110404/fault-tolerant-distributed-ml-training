import pytest

from worker.worker import parse_endpoints


def test_parse_endpoints_assigns_ranks_in_order():
    endpoints = parse_endpoints("worker0:50060, worker1:50061", 2)

    assert endpoints == {
        0: "worker0:50060",
        1: "worker1:50061",
    }


@pytest.mark.parametrize(
    "value",
    ["worker0:50060", "worker0:50060,", "worker0:50060,worker1:50061,worker2:50062"],
)
def test_parse_endpoints_rejects_invalid_cluster_membership(value):
    with pytest.raises(ValueError, match="exactly one"):
        parse_endpoints(value, 2)
