from __future__ import annotations

import ipaddress

from cc_pushback.serve import lan_ip


def test_lan_ip_returns_an_ipv4_address() -> None:
    assert ipaddress.ip_address(lan_ip()).version == 4
