"""End-to-end tests for cross-dataset budget transfers."""
from __future__ import annotations

import asyncio
import dataclasses
import itertools
import os
import tempfile

import aiosqlite
import pytest
import pytest_asyncio

# Point the ledger at an isolated database before importing the app.
_tmp = tempfile.mkdtemp(prefix="dpledger-test-transfer-")
os.environ["DATABASE_PATH"] = os.path.join(_tmp, "transfers.db")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from app import db, rdp  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402

_counter = itertools.count()
_transfer_counter = itertools.count()


def _new_dataset_id(name: str) -> str:
    return f"dsxfer-{name}-{next(_counter)}"


def _new_transfer_id() -> str:
    return f"xfer-{next(_transfer_counter)}"


@pytest_asyncio.fixture()
async def client():
    await db.init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _register(client, dataset_id: str, eps: float = 1.0, delta: float = 1e-6):
    return await client.post(
        f"/datasets/{dataset_id}",
        json={"epsilon_cap": eps, "delta_cap": delta,
              "description": "test dataset"},
    )


async def _transfer(client, transfer_id: str, source: str, target: str,
                    amount: float):
    return await client.post(
        "/budget-transfers",
        json={
            "transfer_id": transfer_id,
            "source_dataset_id": source,
            "target_dataset_id": target,
            "epsilon_amount": amount,
        },
    )


async def _budget(client, dataset_id: str) -> dict:
    r = await client.get(f"/datasets/{dataset_id}/budget")
    assert r.status_code == 200
    return r.json()


# ------------------------------------------------------------- happy path --


@pytest.mark.asyncio
async def test_transfer_moves_cap_and_records_snapshot(client):
    src, dst = _new_dataset_id("src"), _new_dataset_id("dst")
    await _register(client, src, eps=1.0, delta=1e-6)
    await _register(client, dst, eps=0.5, delta=1e-5)

    r = await _transfer(client, _new_transfer_id(), src, dst, 0.25)
    assert r.status_code == 201
    body = r.json()
    assert body["source_dataset_id"] == src
    assert body["target_dataset_id"] == dst
    assert body["epsilon_amount"] == 0.25
    assert body["source_epsilon_cap_before"] == pytest.approx(1.0)
    assert body["source_epsilon_cap_after"] == pytest.approx(0.75)
    assert body["target_epsilon_cap_before"] == pytest.approx(0.5)
    assert body["target_epsilon_cap_after"] == pytest.approx(0.75)
    assert body["source_committed"] == pytest.approx(0.0)
    assert body["target_committed"] == pytest.approx(0.0)
    assert body["idempotent_replay"] is False
    assert body["created_at"]

    src_budget = await _budget(client, src)
    dst_budget = await _budget(client, dst)
    assert src_budget["epsilon_cap"] == pytest.approx(0.75)
    assert dst_budget["epsilon_cap"] == pytest.approx(0.75)
    # delta_cap is never touched by a transfer.
    assert src_budget["delta_cap"] == pytest.approx(1e-6)
    assert dst_budget["delta_cap"] == pytest.approx(1e-5)


@pytest.mark.asyncio
async def test_transfer_leaves_requests_and_delta_untouched(client):
    src, dst = _new_dataset_id("src"), _new_dataset_id("dst")
    await _register(client, src, eps=1.0, delta=1e-6)
    await _register(client, dst, eps=1.0, delta=1e-6)

    granted = await client.post(
        f"/datasets/{src}/requests",
        json={"request_id": "keep-1", "query_type": "histogram",
              "epsilon": 0.2, "delta": 1e-7},
    )
    assert granted.json()["allowed"] is True

    r = await _transfer(client, _new_transfer_id(), src, dst, 0.3)
    assert r.status_code == 201

    # Recorded requests are exactly as before the transfer.
    listing = (await client.get(f"/datasets/{src}/requests")).json()
    assert listing["count"] == 1
    rec = listing["requests"][0]
    assert rec["request_id"] == "keep-1"
    assert rec["status"] == "granted"
    assert rec["epsilon"] == pytest.approx(0.2)

    info = (await client.get(f"/datasets/{src}")).json()
    assert info["budget"]["delta_cap"] == pytest.approx(1e-6)
    assert info["budget"]["granted_requests"] == 1


@pytest.mark.asyncio
async def test_transferred_budget_is_usable_on_target(client):
    src, dst = _new_dataset_id("src"), _new_dataset_id("dst")
    await _register(client, src, eps=1.0, delta=1e-6)
    await _register(client, dst, eps=1.0, delta=1e-6)

    # Pure-DP accounting is exact: 1.2 > 1.0 cap, rejected before transfer.
    big = {"request_id": "big", "query_type": "laplace",
           "epsilon": 1.2, "delta": 0.0}
    before = await client.post(f"/datasets/{dst}/requests", json=big)
    assert before.json()["allowed"] is False

    assert (await _transfer(client, _new_transfer_id(), src, dst, 0.5)).status_code == 201

    after = await client.post(
        f"/datasets/{dst}/requests",
        json={**big, "request_id": "big-2"},
    )
    assert after.json()["allowed"] is True

    # The source now only has 0.5 left: a 0.6 pure-DP request no longer fits.
    too_much = await client.post(
        f"/datasets/{src}/requests",
        json={"request_id": "over", "query_type": "laplace",
              "epsilon": 0.6, "delta": 0.0},
    )
    assert too_much.json()["allowed"] is False


# -------------------------------------------------------------- validation --


@pytest.mark.asyncio
async def test_transfer_source_cap_must_stay_positive(client):
    src, dst = _new_dataset_id("src"), _new_dataset_id("dst")
    await _register(client, src, eps=0.5, delta=1e-6)
    await _register(client, dst, eps=1.0, delta=1e-6)

    for amount in (0.5, 0.6):  # leaves exactly 0.0 / goes negative
        r = await _transfer(client, _new_transfer_id(), src, dst, amount)
        assert r.status_code == 409
        assert r.json()["error"] == "source_cap_not_positive"

    # Both caps unchanged after the rejections.
    assert (await _budget(client, src))["epsilon_cap"] == pytest.approx(0.5)
    assert (await _budget(client, dst))["epsilon_cap"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_transfer_source_must_cover_granted_requests(client):
    src, dst = _new_dataset_id("src"), _new_dataset_id("dst")
    await _register(client, src, eps=1.0, delta=1e-6)
    await _register(client, dst, eps=1.0, delta=1e-6)

    # Pure epsilon-DP: committed is exactly 0.4.
    granted = await client.post(
        f"/datasets/{src}/requests",
        json={"request_id": "r1", "query_type": "laplace",
              "epsilon": 0.4, "delta": 0.0},
    )
    assert granted.json()["allowed"] is True

    # Boundary: leaving exactly the committed 0.4 is allowed.
    ok = await _transfer(client, _new_transfer_id(), src, dst, 0.6)
    assert ok.status_code == 201
    assert ok.json()["source_committed"] == pytest.approx(0.4)

    # Anything more would undercut the committed usage.
    r = await _transfer(client, _new_transfer_id(), src, dst, 0.01)
    assert r.status_code == 409
    assert r.json()["error"] == "source_budget_overdrawn"
    assert (await _budget(client, src))["epsilon_cap"] == pytest.approx(0.4)
    assert (await _budget(client, dst))["epsilon_cap"] == pytest.approx(1.6)


@pytest.mark.asyncio
async def test_transfer_source_must_cover_active_reservations(client):
    src, dst = _new_dataset_id("src"), _new_dataset_id("dst")
    await _register(client, src, eps=1.0, delta=1e-6)
    await _register(client, dst, eps=1.0, delta=1e-6)

    res = await client.post(
        f"/datasets/{src}/reservations",
        json={"reservation_id": "hold", "epsilon": 0.5, "delta": 0.0,
              "ttl_seconds": 3600},
    )
    assert res.json()["allowed"] is True

    # The unexpired reservation occupies 0.5: transferring 0.6 away would
    # leave only 0.4, which does not cover it.
    r = await _transfer(client, _new_transfer_id(), src, dst, 0.6)
    assert r.status_code == 409
    assert r.json()["error"] == "source_budget_overdrawn"

    # Leaving exactly the reserved 0.5 is fine.
    ok = await _transfer(client, _new_transfer_id(), src, dst, 0.5)
    assert ok.status_code == 201
    assert ok.json()["source_committed"] == pytest.approx(0.5)
    assert (await _budget(client, src))["epsilon_cap"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_transfer_target_cap_cannot_exceed_registration_limit(client):
    src, dst = _new_dataset_id("src"), _new_dataset_id("dst")
    await _register(client, src, eps=1.0, delta=1e-6)
    await _register(client, dst, eps=999.5, delta=1e-6)

    r = await _transfer(client, _new_transfer_id(), src, dst, 0.6)
    assert r.status_code == 409
    assert r.json()["error"] == "target_cap_exceeded"

    # Landing exactly on the 1000 limit is allowed.
    ok = await _transfer(client, _new_transfer_id(), src, dst, 0.5)
    assert ok.status_code == 201
    assert (await _budget(client, dst))["epsilon_cap"] == pytest.approx(1000.0)
    assert (await _budget(client, src))["epsilon_cap"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_transfer_rejection_is_all_or_nothing(client):
    src, dst = _new_dataset_id("src"), _new_dataset_id("dst")
    await _register(client, src, eps=1.0, delta=1e-6)
    await _register(client, dst, eps=999.9, delta=1e-6)

    # Fails only because of the target limit; the source side alone would fit.
    r = await _transfer(client, _new_transfer_id(), src, dst, 0.5)
    assert r.status_code == 409
    assert r.json()["error"] == "target_cap_exceeded"
    assert (await _budget(client, src))["epsilon_cap"] == pytest.approx(1.0)
    assert (await _budget(client, dst))["epsilon_cap"] == pytest.approx(999.9)


@pytest.mark.asyncio
async def test_transfer_on_missing_dataset_404(client):
    src = _new_dataset_id("src")
    await _register(client, src, eps=1.0, delta=1e-6)

    r = await _transfer(client, _new_transfer_id(), src, "no-such-dataset", 0.1)
    assert r.status_code == 404
    assert r.json()["error"] == "dataset_not_found"

    r2 = await _transfer(client, _new_transfer_id(), "no-such-dataset", src, 0.1)
    assert r2.status_code == 404
    assert r2.json()["error"] == "dataset_not_found"


@pytest.mark.asyncio
async def test_transfer_payload_validation_422(client):
    src, dst = _new_dataset_id("src"), _new_dataset_id("dst")
    await _register(client, src, eps=1.0, delta=1e-6)
    await _register(client, dst, eps=1.0, delta=1e-6)

    base = {"transfer_id": "t-val", "source_dataset_id": src,
            "target_dataset_id": dst, "epsilon_amount": 0.1}

    for bad in (
        {**base, "epsilon_amount": 0.0},
        {**base, "epsilon_amount": -0.5},
        {**base, "epsilon_amount": "lots"},
        {**base, "transfer_id": ""},
        {**base, "target_dataset_id": src},  # source == target
        {k: v for k, v in base.items() if k != "transfer_id"},
    ):
        r = await client.post("/budget-transfers", json=bad)
        assert r.status_code == 422, bad

    # Nothing was applied while validating.
    assert (await _budget(client, src))["epsilon_cap"] == pytest.approx(1.0)
    assert (await _budget(client, dst))["epsilon_cap"] == pytest.approx(1.0)


# -------------------------------------------------------------- idempotency --


@pytest.mark.asyncio
async def test_transfer_idempotent_replay_returns_original_record(client):
    src, dst = _new_dataset_id("src"), _new_dataset_id("dst")
    await _register(client, src, eps=1.0, delta=1e-6)
    await _register(client, dst, eps=1.0, delta=1e-6)

    tid = _new_transfer_id()
    first = await _transfer(client, tid, src, dst, 0.25)
    assert first.status_code == 201

    for _ in range(3):
        again = await _transfer(client, tid, src, dst, 0.25)
        assert again.status_code == 200
        body = again.json()
        assert body["idempotent_replay"] is True
        # The original record is returned verbatim.
        for key, value in first.json().items():
            if key != "idempotent_replay":
                assert body[key] == value, key

    # Budget moved exactly once.
    assert (await _budget(client, src))["epsilon_cap"] == pytest.approx(0.75)
    assert (await _budget(client, dst))["epsilon_cap"] == pytest.approx(1.25)


@pytest.mark.asyncio
async def test_transfer_same_id_different_payload_conflicts(client):
    src, dst = _new_dataset_id("src"), _new_dataset_id("dst")
    other = _new_dataset_id("other")
    await _register(client, src, eps=1.0, delta=1e-6)
    await _register(client, dst, eps=1.0, delta=1e-6)
    await _register(client, other, eps=1.0, delta=1e-6)

    tid = _new_transfer_id()
    assert (await _transfer(client, tid, src, dst, 0.25)).status_code == 201

    for payload in (
        {"transfer_id": tid, "source_dataset_id": src,
         "target_dataset_id": dst, "epsilon_amount": 0.5},
        {"transfer_id": tid, "source_dataset_id": src,
         "target_dataset_id": other, "epsilon_amount": 0.25},
        {"transfer_id": tid, "source_dataset_id": dst,
         "target_dataset_id": src, "epsilon_amount": 0.25},
    ):
        r = await client.post("/budget-transfers", json=payload)
        assert r.status_code == 409
        assert r.json()["error"] == "transfer_id_conflict"

    # Conflicts never move budget.
    assert (await _budget(client, src))["epsilon_cap"] == pytest.approx(0.75)
    assert (await _budget(client, dst))["epsilon_cap"] == pytest.approx(1.25)
    assert (await _budget(client, other))["epsilon_cap"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_rejected_transfer_id_can_be_retried_after_state_changes(client):
    src, dst = _new_dataset_id("src"), _new_dataset_id("dst")
    await _register(client, src, eps=0.5, delta=1e-6)
    await _register(client, dst, eps=1.0, delta=1e-6)

    tid = _new_transfer_id()
    rejected = await _transfer(client, tid, src, dst, 0.5)
    assert rejected.status_code == 409  # would leave source cap at 0.0

    # A rejected transfer writes no record: once the source has more headroom
    # (here: budget transferred in from elsewhere), the same id succeeds.
    assert (await _transfer(client, _new_transfer_id(), dst, src, 0.4)).status_code == 201
    retried = await _transfer(client, tid, src, dst, 0.5)
    assert retried.status_code == 201
    assert retried.json()["idempotent_replay"] is False


# --------------------------------------------------------------- concurrency --


@pytest.mark.asyncio
async def test_concurrent_identical_transfer_applies_once(client):
    src, dst = _new_dataset_id("src"), _new_dataset_id("dst")
    await _register(client, src, eps=1.0, delta=1e-6)
    await _register(client, dst, eps=1.0, delta=1e-6)

    tid = _new_transfer_id()
    results = await asyncio.gather(*[
        _transfer(client, tid, src, dst, 0.2) for _ in range(20)
    ])
    created = [r for r in results if r.status_code == 201]
    replays = [r for r in results if r.status_code == 200]
    assert len(created) == 1
    assert len(replays) == 19
    assert (await _budget(client, src))["epsilon_cap"] == pytest.approx(0.8)
    assert (await _budget(client, dst))["epsilon_cap"] == pytest.approx(1.2)


@pytest.mark.asyncio
async def test_concurrent_transfers_and_requests_never_overdraw(client):
    src, dst = _new_dataset_id("src"), _new_dataset_id("dst")
    cap = 1.0
    await _register(client, src, eps=cap, delta=1e-6)
    await _register(client, dst, eps=cap, delta=1e-6)

    n_requests, n_transfers = 40, 15
    request_payloads = [
        {"request_id": f"c-{i:03d}", "query_type": "gaussian_mean",
         "epsilon": 0.05, "delta": 1e-8}
        for i in range(n_requests)
    ]
    transfer_ids = [_new_transfer_id() for _ in range(n_transfers)]

    results = await asyncio.gather(
        *[client.post(f"/datasets/{src}/requests", json=p)
          for p in request_payloads],
        *[_transfer(client, tid, src, dst, 0.02) for tid in transfer_ids],
    )
    responses, transfer_responses = results[:n_requests], results[n_requests:]

    ok_transfers = [r for r in transfer_responses if r.status_code == 201]
    rejected_transfers = [r for r in transfer_responses if r.status_code == 409]
    assert len(ok_transfers) + len(rejected_transfers) == n_transfers

    src_budget = await _budget(client, src)
    dst_budget = await _budget(client, dst)

    # Caps moved exactly once per successful transfer.
    assert src_budget["epsilon_cap"] == pytest.approx(cap - 0.02 * len(ok_transfers))
    assert dst_budget["epsilon_cap"] == pytest.approx(cap + 0.02 * len(ok_transfers))

    # The source is never overdrawn, no matter how requests and transfers
    # interleaved.
    granted = [r.json() for r in responses
               if r.status_code in (200, 201) and r.json()["allowed"]]
    curves = [rdp.single_rdp_curve(0.05, 1e-8) for _ in granted]
    consumed, _ = rdp.composed_epsilon(curves, 1e-6)
    assert consumed <= src_budget["epsilon_cap"] + 1e-12
    assert src_budget["epsilon_consumed"] == pytest.approx(consumed, abs=1e-9)
    assert src_budget["epsilon_remaining"] >= -1e-9


# ----------------------------------------------------------------- migration --


@pytest.mark.asyncio
async def test_migration_from_v2_schema(tmp_path, monkeypatch):
    old_db_path = str(tmp_path / "old.db")

    # Build a database at the pre-transfer (v2) schema with some data.
    conn = await aiosqlite.connect(old_db_path)
    try:
        await conn.executescript(db.SCHEMA_V1)
        await conn.executescript(db.SCHEMA_V2)
        await conn.execute(
            "INSERT INTO datasets "
            "(dataset_id, epsilon_cap, delta_cap, description, created_at) "
            "VALUES ('legacy', 1.0, 1e-6, 'old', '2026-01-01T00:00:00+00:00')"
        )
        await conn.execute("PRAGMA user_version = 2")
        await conn.commit()
    finally:
        await conn.close()

    patched = dataclasses.replace(settings, db_path=old_db_path)
    monkeypatch.setattr(db, "settings", patched)
    await db.init_db()

    async with db.connect() as c:
        async with c.execute("PRAGMA user_version") as cur:
            version = (await cur.fetchone())[0]
        assert version == db.LATEST_VERSION
        async with c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='budget_transfers'"
        ) as cur:
            assert await cur.fetchone() is not None
        # Pre-existing data survives the migration.
        async with c.execute(
            "SELECT epsilon_cap FROM datasets WHERE dataset_id = 'legacy'"
        ) as cur:
            row = await cur.fetchone()
        assert row is not None and row[0] == pytest.approx(1.0)
