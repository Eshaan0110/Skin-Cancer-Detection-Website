def test_allowed_file(app_module):
    assert app_module.allowed_file("lesion.jpg")
    assert app_module.allowed_file("lesion.JPEG")
    assert app_module.allowed_file("lesion.png")
    assert not app_module.allowed_file("lesion.gif")
    assert not app_module.allowed_file("lesion")


def test_run_inference_returns_all_classes_ranked(app_module, image_bytes):
    result = app_module.run_inference(image_bytes())

    assert result["label"] in app_module.class_names
    assert result["description"] == app_module.class_descriptions[result["label"]]
    assert result["risk"] == app_module.RISK_LEVELS[result["label"]]
    assert 0.0 <= result["confidence"] <= 100.0

    all_probs = result["all_probabilities"]
    assert len(all_probs) == len(app_module.class_names)
    assert {p["label"] for p in all_probs} == set(app_module.class_names)

    probabilities = [p["probability"] for p in all_probs]
    assert probabilities == sorted(probabilities, reverse=True)
    assert abs(sum(probabilities) - 100.0) < 0.5


def test_run_inference_rejects_unreadable_file(app_module):
    import io
    bad_file = io.BytesIO(b"not an image")
    try:
        app_module.run_inference(bad_file)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "valid image" in str(e)
