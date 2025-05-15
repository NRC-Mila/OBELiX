import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--rundev",
        action="store_true",
        help="Run the dev tests (requires GitPython and access to the private git repository)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "dev: mark test as dev test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--rundev"):
        return
    skip_dev = pytest.mark.skip(
        reason="Dev only. Needs access to private repository to run. Use the --rundev option if you have access."
    )
    for item in items:
        if "dev" in item.keywords:
            item.add_marker(skip_dev)
