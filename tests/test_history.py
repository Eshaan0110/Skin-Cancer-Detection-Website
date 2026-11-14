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


def _post(client, image_bytes, name, path="/api/predict"):
    client.post(path, data={"image": (image_bytes(), name)}, content_type="multipart/form-data")


def test_filter_by_source_and_filename(client, image_bytes):
    _post(client, image_bytes, "alpha.jpg", "/")
    _post(client, image_bytes, "beta.png", "/api/predict")

    data = client.get("/api/history?source=api").get_json()
    assert data["total"] == 1
    assert data["entries"][0]["filename"] == "beta.png"

    data = client.get("/api/history?q=alph").get_json()
    assert [e["filename"] for e in data["entries"]] == ["alpha.jpg"]


def test_filename_search_escapes_wildcards(client, image_bytes):
    _post(client, image_bytes, "alpha.jpg")
    assert client.get("/api/history?q=%25").get_json()["total"] == 0
    assert client.get("/api/history?q=_lpha").get_json()["total"] == 0


def test_invalid_filter_values_are_ignored(client, image_bytes):
    _post(client, image_bytes, "alpha.jpg")
    data = client.get("/api/history?risk=bogus&label=nope&source=x&page=abc").get_json()
    assert data["total"] == 1 and data["page"] == 1


def test_pagination(client, app_module, image_bytes):
    size = app_module.HISTORY_PAGE_SIZE
    for i in range(size + 3):
        _post(client, image_bytes, f"img{i}.jpg")
    p1 = client.get("/api/history").get_json()
    assert p1["pages"] == 2 and len(p1["entries"]) == size
    assert p1["entries"][0]["filename"] == f"img{size + 2}.jpg"
    p2 = client.get("/api/history?page=99").get_json()
    assert p2["page"] == 2 and len(p2["entries"]) == 3
    assert b"Page 1 of 2" in client.get("/history").data


def test_csv_export_respects_filters(client, image_bytes):
    _post(client, image_bytes, "alpha.jpg")
    _post(client, image_bytes, "beta.png")
    resp = client.get("/history/export.csv?q=beta")
    assert resp.mimetype == "text/csv"
    lines = resp.data.decode().strip().splitlines()
    assert lines[0].startswith("id,created_at,filename")
    assert len(lines) == 2 and "beta.png" in lines[1]


def test_empty_filter_result_message(client, image_bytes):
    _post(client, image_bytes, "alpha.jpg")
    assert b"No predictions match these filters" in client.get("/history?q=zzz").data


def test_clear_requires_token_when_configured(client, image_bytes, monkeypatch):
    _post(client, image_bytes, "keep.jpg", "/")
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")

    assert client.post("/history/clear").status_code == 403
    assert client.post("/history/clear", data={"token": "wrong"}).status_code == 403
    assert b"keep.jpg" in client.get("/history").data

    resp = client.post("/history/clear", headers={"X-Admin-Token": "s3cret"})
    assert resp.status_code == 200
    assert b"keep.jpg" not in resp.data


def test_clear_rejects_remote_client_without_token(client, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    resp = client.post("/history/clear", environ_overrides={"REMOTE_ADDR": "203.0.113.5"})
    assert resp.status_code == 403


def test_delete_single_entry_via_api(client, image_bytes):
    _post(client, image_bytes, "one.jpg")
    _post(client, image_bytes, "two.jpg")
    entries = client.get("/api/history").get_json()["entries"]
    target = next(e for e in entries if e["filename"] == "one.jpg")

    resp = client.delete(f"/api/history/{target['id']}")
    assert resp.status_code == 200
    remaining = client.get("/api/history").get_json()
    assert remaining["total"] == 1
    assert remaining["entries"][0]["filename"] == "two.jpg"
    assert client.delete(f"/api/history/{target['id']}").status_code == 404


def test_delete_entry_requires_token_when_configured(client, image_bytes, monkeypatch):
    _post(client, image_bytes, "keep.jpg")
    entry_id = client.get("/api/history").get_json()["entries"][0]["id"]
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")

    assert client.delete(f"/api/history/{entry_id}").status_code == 403
    assert client.post(f"/history/{entry_id}/delete", data={"token": "bad"}).status_code == 403
    assert client.get("/api/history").get_json()["total"] == 1

    resp = client.post(f"/history/{entry_id}/delete", data={"token": "s3cret"})
    assert resp.status_code == 200
    assert client.get("/api/history").get_json()["total"] == 0


def test_delete_entry_rejects_remote_client_without_token(client, image_bytes, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    _post(client, image_bytes, "keep.jpg")
    entry_id = client.get("/api/history").get_json()["entries"][0]["id"]
    resp = client.delete(f"/api/history/{entry_id}", environ_overrides={"REMOTE_ADDR": "203.0.113.5"})
    assert resp.status_code == 403
