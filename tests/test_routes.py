def test_index_get(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_index_post_success(client, image_bytes):
    data = {"image": (image_bytes(), "lesion.jpg")}
    resp = client.post("/", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"risk-badge" in resp.data


def test_index_post_no_file(client):
    resp = client.post("/", data={}, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"No file selected" in resp.data


def test_index_post_bad_extension(client, image_bytes):
    data = {"image": (image_bytes(), "lesion.gif")}
    resp = client.post("/", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"Unsupported file type" in resp.data


def test_api_predict_success(client, image_bytes):
    data = {"image": (image_bytes(), "lesion.png")}
    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body) == {"label", "description", "confidence", "risk", "all_probabilities"}
    assert len(body["all_probabilities"]) == 7


def test_api_predict_no_file(client):
    resp = client.post("/api/predict", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "No file provided."


def test_api_predict_bad_extension(client, image_bytes):
    data = {"image": (image_bytes(), "lesion.bmp")}
    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "Unsupported" in resp.get_json()["error"]


def test_batch_get(client):
    resp = client.get("/batch")
    assert resp.status_code == 200


def test_batch_post_mixed_files(client, image_bytes):
    data = {
        "images": [
            (image_bytes(), "good.jpg"),
            (image_bytes(), "bad.txt"),
        ]
    }
    resp = client.post("/batch", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"good.jpg" in resp.data
    assert b"bad.txt" in resp.data


def test_batch_post_no_files(client):
    resp = client.post("/batch", data={}, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"No files selected" in resp.data


def test_api_batch_success(client, image_bytes):
    data = {
        "images": [
            (image_bytes(), "one.jpg"),
            (image_bytes(), "two.png"),
        ]
    }
    resp = client.post("/api/batch", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["results"]) == 2
    assert all("label" in r for r in body["results"])


def test_api_batch_reports_per_file_errors(client, image_bytes):
    data = {
        "images": [
            (image_bytes(), "good.jpg"),
            (image_bytes(), "bad.exe"),
        ]
    }
    resp = client.post("/api/batch", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    results = {r["filename"]: r for r in resp.get_json()["results"]}
    assert "label" in results["good.jpg"]
    assert results["bad.exe"]["error"] == "Unsupported file type"


def test_api_batch_no_files(client):
    resp = client.post("/api/batch", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
