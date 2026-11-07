import os
import time

import numpy as np
import pytest

from app.pipeline.storage import InferenceStore

CARD = np.zeros((88, 63, 3), np.uint8)


def test_labels_lifecycle(tmp_path):
    store = InferenceStore(tmp_path)
    inference_id = store.save(CARD, {"verdict": "inconclusiva"})
    with pytest.raises(KeyError):
        store.set_label("f" * 32, "real")
    with pytest.raises(ValueError):
        store.set_label(inference_id, "maybe")
    with pytest.raises(ValueError):
        store.set_label("../etc", "real")
    assert store.set_label(inference_id, "real", note="ok")["status"] == "pending"
    with pytest.raises(ValueError):
        store.review(inference_id, "weird")
    assert store.review(inference_id, "rejected", "me")["status"] == "rejected"
    assert list(store.iter_labeled()) == []
    assert list(store.iter_labeled(include_pending=True)) == []  # rejeitado nunca entra


def test_cleanup_respects_labels_and_retention(tmp_path):
    store = InferenceStore(tmp_path)
    old_unlabeled = store.save(CARD, {}, sha256="aa" * 32)
    old_labeled = store.save(CARD, {})
    fresh = store.save(CARD, {})
    store.set_label(old_labeled, "fake")
    ancient = time.time() - 400 * 86400
    for inference_id in (old_unlabeled, old_labeled):
        os.utime(tmp_path / inference_id, (ancient, ancient))

    report = store.cleanup(retention_days=180, max_gb=5)
    assert report["removed"] == 1
    assert not (tmp_path / old_unlabeled).exists()
    assert (tmp_path / old_labeled).exists() and (tmp_path / fresh).exists()
    assert store.find_duplicate("aa" * 32) is None  # ponteiro órfão removido

    # limite de disco: remove as mais antigas sem rótulo até caber
    report = store.cleanup(retention_days=10_000, max_gb=0)
    assert not (tmp_path / fresh).exists() and (tmp_path / old_labeled).exists()


def test_duplicate_index(tmp_path):
    store = InferenceStore(tmp_path)
    inference_id = store.save(CARD, {}, original=b"x", original_format="JPEG", sha256="cd" * 32)
    assert (tmp_path / inference_id / "original.jpg").exists()
    assert store.find_duplicate("cd" * 32) == inference_id
    assert store.find_duplicate("ef" * 32) is None
