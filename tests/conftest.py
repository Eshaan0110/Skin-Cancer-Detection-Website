import io
import os
import sys

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from PIL import Image
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_dummy_checkpoint(path):
    """Save a randomly-initialized model matching app.py's architecture.

    Real torch/torchvision are installed in this environment, so tests exercise
    actual tensor inference end-to-end instead of mocking the model.
    """
    m = efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Linear(m.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 7),
    )
    torch.save(m.state_dict(), path)


@pytest.fixture(scope="session")
def dummy_model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "dummy.pth"
    _build_dummy_checkpoint(str(path))
    return str(path)


@pytest.fixture(scope="session")
def app_module(dummy_model_path, tmp_path_factory):
    os.environ["MODEL_PATH"] = dummy_model_path
    os.environ["HISTORY_DB_PATH"] = str(tmp_path_factory.mktemp("history") / "predictions.db")
    import app as app_module
    assert app_module.model is not None, "dummy checkpoint failed to load"
    return app_module


@pytest.fixture()
def client(app_module):
    app_module.app.config["TESTING"] = True
    app_module.history.clear_history()
    return app_module.app.test_client()


@pytest.fixture()
def image_bytes():
    def _make(fmt="JPEG", size=(64, 64), color=(120, 60, 60)):
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format=fmt)
        buf.seek(0)
        return buf
    return _make
