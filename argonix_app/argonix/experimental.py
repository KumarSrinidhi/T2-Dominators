"""Install the experimental analyzers (trajectory forecasting, fall detection) from the site config.

Both are OFF by default and enabled by the "forecast" / "fall" config keys. Fall detection runs after the threat
analyzers so it can share the altercation analyzer's pose model.
"""
from __future__ import annotations


def install(site) -> list:
    from .fall import install as install_fall
    from .forecast import install as install_forecast
    return [an for an in (install_forecast(site), install_fall(site)) if an is not None]
