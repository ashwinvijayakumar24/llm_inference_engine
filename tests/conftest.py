"""
Shared pytest fixtures — loaded once per session.

scope="session" means weights and oracle data are loaded once for the
entire pytest run, not once per test. Fast re-runs.
"""

import pytest

from engine.loader import load_config, load_weights
from engine.model import LlamaModel
from tests.oracle import load_fixture

WEIGHTS_DIR = "weights"


@pytest.fixture(scope="session")
def config():
    return load_config(WEIGHTS_DIR)


@pytest.fixture(scope="session")
def weights(config):
    return load_weights(WEIGHTS_DIR, config)


@pytest.fixture(scope="session")
def model(weights, config):
    return LlamaModel(weights, config)


@pytest.fixture(scope="session")
def oracle_short():
    return load_fixture("short")


@pytest.fixture(scope="session")
def oracle_medium():
    return load_fixture("medium")
