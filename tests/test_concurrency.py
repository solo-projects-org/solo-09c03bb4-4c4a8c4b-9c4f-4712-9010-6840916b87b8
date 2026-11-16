"""Cross-process concurrency tests: real Uvicorn servers, one database.

These exercise the guarantee that matters in deployment: two web instances
uploading/recomputing at the same time must never publish a detail file that
mixes results from two different compartment versions.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Server:
    def __init__(self, db_path: Path, port: int, cwd: Path):
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        env.update(
            FOREST_DB_PATH=str(db_path),
            FOREST_RUN_WORKER="1",
            FOREST_STALE_RUN_SECONDS="2",
            PYTHONPATH=str(ROOT),
        )
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=cwd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_ready(self, timeout: float = 25.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if httpx.get(self.base + "/healthz", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise RuntimeError("server did not start")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self.proc.kill()


def upload_map(client: httpx.Client, name: str, payload: bytes) -> None:
    response = client.post(
        f"{client.base_url}/maps",
        files={"file": (name, payload, "application/geo+json")},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text


def upload_hotspots(client: httpx.Client, payload: bytes) -> None:
    response = client.post(
        f"{client.base_url}/batches",
        files={"file": ("hotspots.csv", payload, "text/csv")},
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text


def wait_published(client: httpx.Client, batch_id: int, timeout: float = 25.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"{client.base_url}/api/batches/{batch_id}").json()
        if payload["published_total"] is not None and not payload["pending"]:
            return payload
        time.sleep(0.05)
    raise AssertionError("batch never published")


@pytest.mark.slow
def test_two_servers_publish_single_consistent_version(tmp_path):
    db_path = tmp_path / "forest.db"
    port_a, port_b = free_port(), free_port()
    server_a = Server(db_path, port_a, tmp_path)
    server_b = Server(db_path, port_b, tmp_path)
    server_a.wait_ready()
    server_b.wait_ready()
    try:
        with httpx.Client(base_url=server_a.base, timeout=30) as client_a, httpx.Client(
            base_url=server_b.base, timeout=30
        ) as client_b:
            upload_map(client_a, "v1.geojson", (SAMPLES / "compartments_v1.geojson").read_bytes())
            upload_hotspots(client_b, (SAMPLES / "hotspots.csv").read_bytes())
            first = wait_published(client_a, 1)
            assert first["published_map_version"] == 1

            # Instance A swaps the map while instance B triggers a recompute.
            upload_map(client_a, "v2.geojson", (SAMPLES / "compartments_v2.geojson").read_bytes())
            response = client_b.post(
                f"{client_b.base_url}/batches/1/recompute", follow_redirects=False
            )
            assert response.status_code == 303

            final = wait_published(client_b, 1)
            assert final["published_map_version"] == 2
            assert final["published_total"] == 15

            # The published CSV must be internally consistent: it is produced
            # exclusively from the run pinned to the active v2 map.
            csv_bytes = client_a.get(f"{client_a.base_url}/batches/1/download.csv").content
            text = csv_bytes.decode("utf-8-sig")
            rows = text.splitlines()
            assert len(rows) == 16
            assert all("v2" in line for line in rows[1:])
            assert "v1" not in text
            assert "A-01;B-02" not in text  # v1-only shared edge between A and B

            # Every hotspot row present exactly once.
            assert text.count("H-00") == 15

            # No computation against the retired v1 map may remain pending, and
            # no v1 run may be the published one.
            payload = client_a.get(f"{client_a.base_url}/api/batches/1").json()
            active_version = client_a.get(f"{client_a.base_url}/healthz").json()[
                "active_map_version"
            ]
            assert payload["published_map_version"] == active_version == 2
            for run in payload["runs"]:
                if run["map_version_id"] == 1:
                    assert run["status"] in {"succeeded", "superseded"}
            published_runs = [
                run for run in payload["runs"] if run["id"] == payload["published_run_id"]
            ]
            assert published_runs[0]["map_version_id"] == active_version
    finally:
        server_a.stop()
        server_b.stop()


# --------------------------------------------------------------------------
# Conflict review across two real servers
# --------------------------------------------------------------------------


def _review_page_rows(client: httpx.Client, batch_id: int = 1):
    """Parse the review forms rendered on the batch page (server-side HTML)."""

    import html as html_module
    import re as re_module

    page = client.get(f"{client.base_url}/batches/{batch_id}").text
    rows = []
    for block in page.split('<div class="review-item"')[1:]:
        seq = int(re_module.search(r'id="conflict-(\d+)"', block).group(1))
        run_id = int(re_module.search(r'name="run_id" value="(\d+)"', block).group(1))
        revision = int(re_module.search(r'name="revision" value="(\d+)"', block).group(1))
        candidates = re_module.findall(r'<option value="([^"]+)"', block)
        rows.append(
            {
                "seq": seq,
                "run_id": run_id,
                "revision": revision,
                "candidates": [html_module.unescape(item) for item in candidates],
            }
        )
    return rows


def _post_decision(client: httpx.Client, row, code, reason, batch_id: int = 1):
    return client.post(
        f"{client.base_url}/batches/{batch_id}/review/{row['seq']}",
        data={
            "run_id": str(row["run_id"]),
            "revision": str(row["revision"]),
            "code": code,
            "reason": reason,
        },
        follow_redirects=False,
    )


@pytest.mark.slow
def test_two_servers_review_workflow(tmp_path):
    """Adjudications, optimistic conflicts and the reviewed file span processes."""

    db_path = tmp_path / "forest.db"
    port_a, port_b = free_port(), free_port()
    server_a = Server(db_path, port_a, tmp_path)
    server_b = Server(db_path, port_b, tmp_path)
    server_a.wait_ready()
    server_b.wait_ready()
    try:
        with httpx.Client(base_url=server_a.base, timeout=30) as client_a, httpx.Client(
            base_url=server_b.base, timeout=30
        ) as client_b:
            upload_map(client_a, "v1.geojson", (SAMPLES / "compartments_v1.geojson").read_bytes())
            upload_hotspots(client_b, (SAMPLES / "hotspots.csv").read_bytes())
            first = wait_published(client_a, 1)
            assert first["conflict"] == 6

            rows = _review_page_rows(client_b)
            assert len(rows) == 6

            # Gate holds on the instance that did not do the uploading.
            blocked = client_b.get(
                f"{client_b.base_url}/batches/1/download_review.csv", follow_redirects=False
            )
            assert blocked.status_code == 303
            assert "err=" in blocked.headers["location"]

            # Both instances rendered the same revision of the same row.
            row = rows[0]
            first_write = _post_decision(client_a, row, row["candidates"][0], "甲实例核实")
            assert "err=" not in first_write.headers["location"]
            late_write = _post_decision(client_b, row, row["candidates"][-1], "乙实例核实")
            assert "err=" in late_write.headers["location"]
            from urllib.parse import unquote

            assert "已被其他审核员修改" in unquote(late_write.headers["location"])

            # Decide the remaining conflicts from whichever instance renders them.
            for index, current in enumerate(_review_page_rows(client_b)):
                if current["seq"] == row["seq"]:
                    continue
                response = _post_decision(
                    client_b,
                    current,
                    current["candidates"][0],
                    f"并发审核 {index}",
                )
                assert "err=" not in response.headers["location"], response.headers["location"]

            reviewed = client_a.get(f"{client_a.base_url}/batches/1/download_review.csv")
            assert reviewed.status_code == 200
            assert "attachment" in reviewed.headers["content-disposition"]
            import csv
            import io

            parsed = list(csv.DictReader(io.StringIO(reviewed.content.decode("utf-8-sig"))))
            assert len(parsed) == 15
            assert all(item["计算记录号"] == f"run-{first['published_run_id']}" for item in parsed)
            conflict_rows = [item for item in parsed if item["自动归属状态"] == "归属冲突"]
            assert len(conflict_rows) == 6
            assert all(item["裁定林班号"] and item["裁定理由"] for item in conflict_rows)
            assert all(item["自动归属状态"] for item in parsed)

            # A recompute on one instance must demand fresh decisions, while the
            # old ones remain attached to the old run.
            client_b.post(f"{client_b.base_url}/batches/1/recompute", follow_redirects=False)
            second = wait_published(client_a, 1)
            assert second["published_run_id"] != first["published_run_id"]
            rows_after = _review_page_rows(client_a)
            assert len(rows_after) == second["conflict"]
            assert all(item["revision"] == 0 for item in rows_after)
            assert all(item["run_id"] == second["published_run_id"] for item in rows_after)
            blocked = client_a.get(
                f"{client_a.base_url}/batches/1/download_review.csv", follow_redirects=False
            )
            assert "err=" in blocked.headers["location"]
    finally:
        server_a.stop()
        server_b.stop()


@pytest.mark.slow
def test_concurrent_map_swaps_leave_exactly_one_active(tmp_path):
    db_path = tmp_path / "forest.db"
    port_a, port_b = free_port(), free_port()
    server_a = Server(db_path, port_a, tmp_path)
    server_b = Server(db_path, port_b, tmp_path)
    server_a.wait_ready()
    server_b.wait_ready()
    try:
        with httpx.Client(base_url=server_a.base, timeout=30) as client_a, httpx.Client(
            base_url=server_b.base, timeout=30
        ) as client_b:
            upload_hotspots(client_a, (SAMPLES / "hotspots.csv").read_bytes())
            v1 = (SAMPLES / "compartments_v1.geojson").read_bytes()
            v2 = (SAMPLES / "compartments_v2.geojson").read_bytes()

            import threading

            errors: list[BaseException] = []

            def swap(client, payload, name):
                try:
                    upload_map(client, name, payload)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [
                threading.Thread(target=swap, args=(client_a, v1, "a.geojson")),
                threading.Thread(target=swap, args=(client_b, v2, "b.geojson")),
                threading.Thread(target=swap, args=(client_a, v2, "c.geojson")),
                threading.Thread(target=swap, args=(client_b, v1, "d.geojson")),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert not errors

            health = client_a.get(f"{client_a.base_url}/healthz").json()
            assert health["active_map_version"] is not None
            from app import db, service

            with db.connection(str(db_path)) as conn:
                active_count = conn.execute(
                    "SELECT COUNT(*) AS n FROM map_versions WHERE is_active = 1"
                ).fetchone()["n"]
                versions = conn.execute("SELECT COUNT(*) AS n FROM map_versions").fetchone()["n"]
            assert active_count == 1
            assert versions == 4

            # The active version is the one the health endpoint reports, and a
            # recompute publishes against it with no mixed output.
            client_b.post(f"{client_b.base_url}/batches/1/recompute", follow_redirects=False)
            final = wait_published(client_b, 1)
            assert final["published_map_version"] == health["active_map_version"]
    finally:
        server_a.stop()
        server_b.stop()
