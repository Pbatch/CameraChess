import pytest
from loguru import logger


@pytest.fixture
def log_messages():
    """What loguru logs during the test, each as '<LEVEL> <message>'"""
    messages: list[str] = []
    handler = logger.add(messages.append, format="{level} {message}")
    yield messages
    logger.remove(handler)
