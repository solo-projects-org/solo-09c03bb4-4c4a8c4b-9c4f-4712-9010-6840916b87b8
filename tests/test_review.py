"""Reviewer adjudication of automatic conflicts.

Covers the review workflow end to end:

* only the published run that matches the active map is reviewable;
* every conflict must be adjudicated before the reviewed CSV can be produced;
* two reviewers who loaded the same revision cannot silently overwrite each
  other — the late submission sees a conflict notice;
* decisions stay attached to their computation after a recompute/map swap;
* a reviewed download is one snapshot even if another instance changes
  decisions or republishes mid-stream.
"""

from __future__ import annotations

import csv
import io
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db, service
from app.main import create_app
from app.worker import Worker

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def make_client(tmp_path, name="forest.db"):
    app = create_app(db_path=str(tmp_path / name), start_worker=False)
    return TestClient(app), app


def drain(app):
    while True:
        claimed = service.claim_next_run(app.state.settings["db_path"])
        if claimed is None:
            return
        run_id, token = claimed
        service.execute_run(app.state.settings["db_path"], run_id, attempt_token=token)


def upload_samples(client, map_name="compartments_v1.geojson"):
    client.post(
        "/maps",
        files={"file": (map_name, (SAMPLES / map_name).read_bytes(), "application/geo+json")},
    )
    client.post(
        "/batches",
        files={"file": ("hotspots.csv", (SAMPLES / "hotspots.csv").read_bytes(), "text/csv")},
    )


def review_rows(app, batch_id=1):
    with db.connection(app.state.settings["db_path"]) as c:
        batch = service.get_batch(c, batch_id)
        run = service.current_review_run(c, batch)
        assert run is not None
        return run, service.list_review_rows(c, int(run["id"]))


def submit(client, batch_id, row, code, reason, revision=None, run_id=None):
    return client.post(
        f"/batches/{batch_id}/review/{row['hotspot_seq']}",
        data={
            "run_id": str(run_id if run_id is not None else row["run_id"]),
            "revision": str(row["revision"] if revision is None else revision),
            "code": code,
            "reason": reason,
        },
        follow_redirects=False,
    )


def decide_all(client, app, batch_id=1, suffix=""):
    run, rows = review_rows(app, batch_id)
    assert rows
    for row in rows:
        response = submit(
            client,
            batch_id,
            row,
            row["candidates"][0],
            f"现场核实{suffix}".strip(),
            revision=0,
            run_id=run["id"],
        )
        assert response.status_code == 303
        assert "err=" not in response.headers["location"]
    return run, rows


def parse_csv(content: bytes):
    text = content.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def test_review_gate_and_reviewed_csv(tmp_path):
    client, app = make_client(tmp_path)
    with client:
        upload_samples(client)
        drain(app)
        run, rows = review_rows(app, 1)
        assert len(rows) == 6
        assert all(row["decision"] is None for row in rows)

        # Gate: not all conflicts adjudicated -> no reviewed file.
        blocked = client.get("/batches/1/download_review.csv", follow_redirects=False)
        assert blocked.status_code == 303
        assert "err=" in blocked.headers["location"]
        assert "download_review.csv" not in blocked.headers.get("location", "")

        decided_rows = []
        for row in rows:
            chosen = row["candidates"][-1]
            response = submit(
                client, 1, row, chosen, f"核实后归 {chosen}", revision=0, run_id=run["id"]
            )
            assert response.status_code == 303
            assert "err=" not in response.headers["location"], response.headers["location"]
            decided_rows.append((row["source_row"], chosen, row["candidates"]))

        ok = client.get("/batches/1/download_review.csv")
        assert ok.status_code == 200
        assert "attachment" in ok.headers["content-disposition"]
        rows_out = parse_csv(ok.content)
        assert len(rows_out) == 15
        by_row = {int(item["来源行号"]): item for item in rows_out}
        for source_row, chosen, candidates in decided_rows:
            item = by_row[source_row]
            assert item["自动归属状态"] == "归属冲突"
            assert item["候选林班号"] == ";".join(candidates)
            assert item["裁定林班号"] == chosen
            assert item["最终林班号"] == chosen
            assert item["裁定理由"] == f"核实后归 {chosen}"
            assert item["裁定版本"] == "r1"
            assert item["计算记录号"] == f"run-{run['id']}"
        matched = [item for item in rows_out if item["自动归属状态"] == "已赋码"]
        unmatched = [item for item in rows_out if item["自动归属状态"] == "未匹配"]
        assert matched and all(item["自动归属林班号"] for item in matched)
        assert matched[0]["最终林班号"] == matched[0]["自动归属林班号"]
        assert unmatched and all(item["最终林班号"] == "" for item in unmatched)
        # Automatic attribution is preserved for every conflict row too.
        assert all(item["自动归属林班号"] == "" for item in rows_out if item["自动归属状态"] == "归属冲突")


def test_two_reviewers_same_revision_late_one_sees_conflict(tmp_path):
    """Both loaded revision 0; the second submit must not overwrite the first."""

    client_a, app_a = make_client(tmp_path)
    client_b, app_b = make_client(tmp_path)
    with client_a, client_b:
        upload_samples(client_a)
        drain(app_a)
        run, rows = review_rows(app_a, 1)
        row = rows[0]
        # Both reviewers rendered the same row at revision 0.
        first = submit(client_a, 1, row, row["candidates"][0], "审核员甲：A 现场核实", revision=0, run_id=run["id"])
        assert "err=" not in first.headers["location"]
        late = submit(client_b, 1, row, row["candidates"][1], "审核员乙：B 现场核实", revision=0, run_id=run["id"])
        assert late.status_code == 303
        location = late.headers["location"]
        assert "err=" in location
        # The message names the winning decision so the reviewer can re-read it.
        from urllib.parse import unquote

        assert unquote(location).count("已被其他审核员修改") == 1
        assert row["candidates"][0] in unquote(location)

        # Stored decision is still the first one.
        with db.connection(app_a.state.settings["db_path"]) as c:
            stored = service.get_decision(c, int(run["id"]), row["hotspot_seq"])
        assert stored["code"] == row["candidates"][0]
        assert stored["reason"] == "审核员甲：A 现场核实"
        assert stored["revision"] == 1

        # Reloading and submitting from the fresh revision succeeds.
        _, refreshed = review_rows(app_a, 1)
        current = next(item for item in refreshed if item["hotspot_seq"] == row["hotspot_seq"])
        assert current["decision"]["revision"] == 1
        assert current["decision"]["code"] == row["candidates"][0]
        again = submit(
            client_b, 1, current, row["candidates"][1], "审核员乙：改判 B", revision=1, run_id=run["id"]
        )
        assert "err=" not in again.headers["location"]
        with db.connection(app_a.state.settings["db_path"]) as c:
            stored = service.get_decision(c, int(run["id"]), row["hotspot_seq"])
        assert stored["code"] == row["candidates"][1]
        assert stored["revision"] == 2


def test_concurrent_reviewer_submissions_have_one_winner(tmp_path):
    """Real parallel submissions on two app instances: exactly one r1 wins."""

    client_a, app_a = make_client(tmp_path)
    client_b, app_b = make_client(tmp_path)
    with client_a, client_b:
        upload_samples(client_a)
        drain(app_a)
        run, rows = review_rows(app_a, 1)
        row = rows[0]
        results = []

        def run_submit(client, code, reason):
            response = submit(client, 1, row, code, reason, revision=0, run_id=run["id"])
            results.append((code, "err=" not in response.headers["location"]))

        threads = [
            threading.Thread(target=run_submit, args=(client_a, row["candidates"][0], "甲")),
            threading.Thread(target=run_submit, args=(client_b, row["candidates"][1], "乙")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted(ok for _, ok in results) == [False, True]
        with db.connection(app_a.state.settings["db_path"]) as c:
            stored = service.get_decision(c, int(run["id"]), row["hotspot_seq"])
        winner = next(code for code, ok in results if ok)
        assert stored["code"] == winner
        assert stored["revision"] == 1


def test_decision_must_be_a_candidate_and_reason_required(tmp_path):
    client, app = make_client(tmp_path)
    with client:
        upload_samples(client)
        drain(app)
        run, rows = review_rows(app, 1)
        row = rows[0]
        bad = submit(client, 1, row, "Z-99", "不属于候选", revision=0, run_id=run["id"])
        assert "err=" in bad.headers["location"]
        blank = submit(client, 1, row, row["candidates"][0], "   ", revision=0, run_id=run["id"])
        assert "err=" in blank.headers["location"]
        with db.connection(app.state.settings["db_path"]) as c:
            assert service.get_decision(c, int(run["id"]), row["hotspot_seq"]) is None


def test_matched_and_unmatched_rows_cannot_be_decided(tmp_path):
    client, app = make_client(tmp_path)
    with client:
        upload_samples(client)
        drain(app)
        run, _ = review_rows(app, 1)
        with db.connection(app.state.settings["db_path"]) as c:
            matched = c.execute(
                "SELECT hotspot_seq FROM results WHERE run_id=? AND status='matched' LIMIT 1",
                (run["id"],),
            ).fetchone()
        fake = {"hotspot_seq": matched["hotspot_seq"], "revision": 0}
        response = submit(client, 1, fake, "A-01", "不应允许", revision=0, run_id=run["id"])
        assert "err=" in response.headers["location"]


def test_recompute_keeps_old_decisions_with_old_run(tmp_path):
    client, app = make_client(tmp_path)
    with client:
        upload_samples(client)
        drain(app)
        old_run, rows = decide_all(client, app, suffix="旧底图")
        assert client.get("/batches/1/download_review.csv").status_code == 200

        # Swap the map; until recompute there is no current review target.
        client.post(
            "/maps",
            files={
                "file": (
                    "compartments_v2.geojson",
                    (SAMPLES / "compartments_v2.geojson").read_bytes(),
                    "application/geo+json",
                )
            },
        )
        page = client.get("/batches/1")
        assert page.status_code == 200
        assert "按当前底图重算" in page.text
        blocked = client.get("/batches/1/download_review.csv", follow_redirects=False)
        assert "err=" in blocked.headers["location"]

        client.post("/batches/1/recompute")
        drain(app)
        new_run, new_rows = review_rows(app, 1)
        assert new_run["id"] != old_run["id"]
        assert new_run["map_version_id"] == 2
        assert all(row["decision"] is None for row in new_rows)

        # Old decisions remain stored against the old run and are shown in history.
        with db.connection(app.state.settings["db_path"]) as c:
            old_progress = service.review_progress(c, int(old_run["id"]))
            history = service.historical_decisions(c, 1)
        assert old_progress["complete"] is True
        assert old_progress["decided"] == len(rows) == 6
        assert {item["run_id"] for item in history} == {old_run["id"]}
        page = client.get("/batches/1")
        assert "历史裁定" in page.text

        # The new computation must be adjudicated from scratch before export.
        blocked = client.get("/batches/1/download_review.csv", follow_redirects=False)
        assert "err=" in blocked.headers["location"]
        decide_all(client, app, suffix="新底图")
        fresh = parse_csv(client.get("/batches/1/download_review.csv").content)
        assert all(item["计算记录号"] == f"run-{new_run['id']}" for item in fresh)
        assert all(item["裁定理由"] == "现场核实新底图" for item in fresh if item["自动归属状态"] == "归属冲突")


def test_decision_rejected_if_run_is_no_longer_published(tmp_path):
    """A form rendered for an old run cannot write a decision after republish."""

    client, app = make_client(tmp_path)
    with client:
        upload_samples(client)
        drain(app)
        old_run, old_rows = review_rows(app, 1)
        stale_row = old_rows[0]

        client.post("/batches/1/recompute")
        drain(app)
        new_run, _ = review_rows(app, 1)
        assert new_run["id"] != old_run["id"]

        response = submit(
            client, 1, stale_row, stale_row["candidates"][0], "旧表单", revision=0, run_id=old_run["id"]
        )
        assert "err=" in response.headers["location"]
        with db.connection(app.state.settings["db_path"]) as c:
            assert service.get_decision(c, int(old_run["id"]), stale_row["hotspot_seq"]) is None
            # The old run itself is no longer reviewable through the API either.
            with pytest.raises(service.ReviewConflict):
                service.save_decision(
                    app.state.settings["db_path"],
                    1,
                    int(old_run["id"]),
                    stale_row["hotspot_seq"],
                    stale_row["candidates"][0],
                    "旧计算",
                    0,
                )


def test_reviewed_download_is_one_snapshot(tmp_path):
    """Mutating decisions mid-stream must not leak into the running export."""

    client, app = make_client(tmp_path)
    db_path = app.state.settings["db_path"]
    with client:
        upload_samples(client)
        drain(app)
        run, rows = decide_all(client, app, suffix="快照")

        filename, generator = service.begin_reviewed_export(db_path, 1, chunk=1)
        assert filename.endswith(f"run{run['id']}.csv")
        header = next(generator)
        first_data = next(generator)
        assert "自动归属状态" in header

        # Another instance changes a decision and a new computation publishes
        # while the download is in flight.
        target = rows[0]
        service.save_decision(
            db_path, 1, int(run["id"]), target["hotspot_seq"],
            target["candidates"][-1], "下载期间改判", 1,
        )
        client.post("/batches/1/recompute")
        drain(app)
        new_run, _ = review_rows(app, 1)
        assert new_run["id"] != run["id"]

        rest = "".join(generator)
        content = (header + first_data + rest).encode()
        parsed = parse_csv_bom(content)
        assert len(parsed) == 15
        # Every row still belongs to the run pinned at export start.
        assert all(item["计算记录号"] == f"run-{run['id']}" for item in parsed)
        # And the decision set is the one from the snapshot (r1, old reason).
        for item in parsed:
            if item["自动归属状态"] == "归属冲突":
                assert item["裁定版本"] == "r1"
                assert item["裁定理由"] == "现场核实快照"


def parse_csv_bom(content: bytes):
    return list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))


def test_reviewed_export_rejects_stale_map_without_recompute(tmp_path):
    client, app = make_client(tmp_path)
    with client:
        upload_samples(client)
        drain(app)
        decide_all(client, app)
        client.post(
            "/maps",
            files={
                "file": (
                    "compartments_v2.geojson",
                    (SAMPLES / "compartments_v2.geojson").read_bytes(),
                    "application/geo+json",
                )
            },
        )
        with pytest.raises(service.ServiceError):
            service.begin_reviewed_export(app.state.settings["db_path"], 1)


def test_decision_rejected_after_map_swap_before_recompute(tmp_path):
    """The published run is unchanged, but its map is no longer current."""

    client, app = make_client(tmp_path)
    with client:
        upload_samples(client)
        drain(app)
        old_run, rows = review_rows(app, 1)
        row = rows[0]
        client.post(
            "/maps",
            files={
                "file": (
                    "compartments_v2.geojson",
                    (SAMPLES / "compartments_v2.geojson").read_bytes(),
                    "application/geo+json",
                )
            },
        )
        response = submit(client, 1, row, row["candidates"][0], "换图后提交", revision=0, run_id=old_run["id"])
        assert "err=" in response.headers["location"]
        with db.connection(app.state.settings["db_path"]) as c:
            assert service.get_decision(c, int(old_run["id"]), row["hotspot_seq"]) is None
            assert service.current_review_run(c, service.get_batch(c, 1)) is None


def test_worker_publishes_and_review_target_follows_publication(tmp_path):
    """With a live worker, review is only offered once the new run is published."""

    client, app = make_client(tmp_path)
    db_path = app.state.settings["db_path"]
    with client:
        upload_samples(client)
        worker = Worker(db_path, poll_interval=0.02)
        worker.start()
        try:
            import time

            deadline = time.time() + 15
            while time.time() < deadline:
                payload = client.get("/api/batches/1").json()
                if payload["published_total"] is not None and not payload["pending"]:
                    break
                time.sleep(0.05)
            else:
                pytest.fail("worker did not publish")
        finally:
            worker.stop()
            worker.join(timeout=5)
        page = client.get("/batches/1")
        assert "已裁定 0 / 6" in page.text
        payload = client.get("/api/batches/1").json()
        assert payload["review"]["conflict_total"] == 6
        assert payload["review"]["decided"] == 0
        assert payload["review"]["complete"] is False


def test_zero_conflict_batch_can_export_reviewed_csv(tmp_path):
    """No conflicts means the gate is trivially satisfied."""

    client, app = make_client(tmp_path)
    with client:
        client.post(
            "/maps",
            files={"file": ("m.geojson", geojson({"A": (0, 0, 1, 1)}), "application/geo+json")},
        )
        client.post(
            "/batches",
            files={"file": ("h.csv", csv_bytes([("H1", 0.5, 0.5, "2026-01-01T00:00:00Z")]), "text/csv")},
        )
        drain(app)
        page = client.get("/batches/1")
        assert "没有自动归属冲突" in page.text
        response = client.get("/batches/1/download_review.csv")
        assert response.status_code == 200
        rows = parse_csv(response.content)
        assert len(rows) == 1
        assert rows[0]["自动归属状态"] == "已赋码"
        assert rows[0]["最终林班号"] == "A"
        assert rows[0]["裁定林班号"] == ""


def geojson(polygons):
    import json

    features = []
    for code, (minx, miny, maxx, maxy) in polygons.items():
        features.append(
            {
                "type": "Feature",
                "properties": {"code": code},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
                    ],
                },
            }
        )
    return json.dumps({"type": "FeatureCollection", "features": features}).encode()


def csv_bytes(rows):
    lines = ["hotspot_no,lon,lat,observed_at"]
    lines += [",".join(map(str, row)) for row in rows]
    return ("\n".join(lines) + "\n").encode()
