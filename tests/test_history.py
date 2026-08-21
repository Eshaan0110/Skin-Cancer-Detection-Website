def test_history_empty(client):
    resp = client.get("/history")
    assert resp.status_code == 200
    assert b"No predictions logged yet" in resp.data


def test_single_prediction_appears_in_history(client, image_bytes):
    data = {"image": (image_bytes(), "mole.jpg")}
    client.post("/", data=data, content_type="multipart/form-data")

    resp = client.get("/history")
    assert resp.status_code == 200
    assert b"mole.jpg" in resp.data
    assert b"web</td>" in resp.data


def test_api_predict_logs_with_api_source(client, image_bytes):
    data = {"image": (image_bytes(), "api-lesion.png")}
    client.post("/api/predict", data=data, content_type="multipart/form-data")

    resp = client.get("/history")
    assert b"api-lesion.png" in resp.data
    assert b"api</td>" in resp.data


def test_batch_predictions_all_logged(client, image_bytes):
    data = {
        "images": [
            (image_bytes(), "batch-one.jpg"),
            (image_bytes(), "batch-two.png"),
        ]
    }
    client.post("/batch", data=data, content_type="multipart/form-data")

    resp = client.get("/history")
    assert b"batch-one.jpg" in resp.data
    assert b"batch-two.png" in resp.data
    assert b"web-batch</td>" in resp.data


def test_invalid_files_are_not_logged(client, image_bytes):
    data = {"image": (image_bytes(), "lesion.gif")}
    client.post("/", data=data, content_type="multipart/form-data")

    resp = client.get("/history")
    assert b"lesion.gif" not in resp.data


def test_risk_counts_reflect_logged_predictions(client, app_module, image_bytes):
    data = {"image": (image_bytes(), "mole.jpg")}
    client.post("/", data=data, content_type="multipart/form-data")

    counts = app_module.history.get_risk_counts()
    assert sum(counts.values()) == 1


def test_history_clear_removes_entries(client, image_bytes):
    data = {"image": (image_bytes(), "mole.jpg")}
    client.post("/", data=data, content_type="multipart/form-data")
    assert b"mole.jpg" in client.get("/history").data

    resp = client.post("/history/clear")
    assert resp.status_code == 200
    assert b"No predictions logged yet" in resp.data
