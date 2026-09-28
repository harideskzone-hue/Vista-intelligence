# Border Vigil AI (SIH26187) — End-to-End Implementation Plan

**Reference architecture:** `SIH26187_Border_Surveillance_Architecture_v5_FINAL.mermaid`
**Problem statement:** SIH26187, Ministry of Home Affairs — edge-first AI video analytics for border surveillance + check-post identity/vehicle screening

This plan turns the v5 target architecture into an executable build sequence. Every phase names the exact diagram nodes it implements, so the team can work directly off the mermaid file instead of a separate spec that can drift out of sync.

---

## 0. How to read this plan

- **Phases are sequential in dependency, not necessarily in calendar time.** Phases 1–2 (data + core AI/ML) can run in parallel across team members from day one. Phases 4–5 (backend + dashboards) can start as soon as Phase 3's event schema is frozen, even before Phase 3's logic is fully correct.
- **MVP vs Target scope** — the v5 diagram deliberately includes Phase 2/3 boxes (dashed/`future` styled: `FORCE_C2`, `CIBMS`, `VAHAN`, `UIDAI`, `OTHER_AG`, `CLOUD`, `LEGAL`, `FEEDBACK`) that this plan does **not** build. Anything in that style class is out of scope for the hackathon build. If in doubt, check the node's classDef in the mermaid file before spending time on it.
- **"Measured, not claimed"** is the standing rule for this project (it's baked into the `THRESH` node itself). No phase is "done" until its accuracy/latency claim is backed by a number your team actually produced, not a number that sounds right.

---

## Phase 0 — Project Setup & Scope Lock

**Objective:** Everyone is building against the same frozen scope and the same repo, before any AI/ML code is written.

Tasks:
- Walk the whole team through the v5 mermaid diagram once, live. Confirm everyone agrees on what's MVP vs Phase 2/3.
- Freeze the **MVP component list** in writing (copy the table below into your repo's `SCOPE.md`, edit as needed):

| Layer | In MVP | Deferred |
|---|---|---|
| Edge | Perimeter CCTV via RTSP/ONVIF, Check-post face + ANPR cameras | Multi-site fleet management |
| AI/ML | YOLO26-N + ByteTrack (perimeter), SCRFD + ArcFace (face), YOLO26-N + PaddleOCR (ANPR) | Active-learning retraining |
| Fusion/Decision | Event Trigger → Signal-to-Event Association → PV correlation → Risk + Policy → Final Alert | — (this is the core deliverable, not deferrable) |
| Backend/Data | Event API, Alert Service, Event DB, Evidence Store, Audit Log | Sovereign cloud hosting (`CLOUD`) |
| C2 | Operator dashboard, Officer console, Central rollup (simulated 2–3 BOPs) | Real Force C2 (`FORCE_C2`), CIBMS interconnect |
| External | — | UIDAI, VAHAN, other agencies (all Phase 2/3 MoU-gated) |
| Governance | Camera config, threshold calibration, GPU→CPU fallback, retention+expiry, RBAC | Legal/chain-of-custody system |

- Repo structure: separate top-level dirs for `edge/` (perimeter + face + ANPR pipelines), `fusion/` (event engine), `backend/` (API + services), `dashboard/`, `data/` (datasets, DB schema, seed scripts), `docs/` (this plan + architecture file live here).
- Set up environments: a CPU-only dev container that mirrors the `CPU_FALLBACK` path (so everyone can develop without needing a Jetson on their desk), plus one Jetson Orin (or cloud GPU instance) for realistic latency testing.
- Stand up CI: lint + unit tests on push, at minimum.
- Assign rough ownership by layer (Edge/CV, Fusion/Decision logic, Backend/Data, Dashboard/Frontend, Data & Evaluation) — even a small team should have a named owner per layer so the Phase 3 fusion logic doesn't become an orphan nobody is responsible for.

**Exit criteria:** `SCOPE.md` committed and agreed by the whole team; repo skeleton + CI green; every team member can run the CPU dev container.

---

## Phase 1 — Data Ingestion & Dataset Preparation

**Implements:** `CCTV_PERI`, `INGEST`, `CCTV_BCP`, `FACE_CAM`, `ANPR_CAM`, `CAM_CONFIG`, `AUTH_DB` seed data, `EVAL_DATASET` + `GROUND_TRUTH`

Tasks:
- **Perimeter data:** source or record demo footage covering zone/line-crossing scenarios (person, vehicle, animal/vegetation false-alarm cases for `FA_FILTER` training). Public datasets (e.g. VIRAT, UCF-Crime-adjacent open sets) plus your own recorded clips for the specific demo zone.
- **Face data:** build an authorized-personnel enrollment set (consented team/volunteer photos — do **not** scrape real people's biometric data without consent, even for a demo) plus a public face dataset for model validation (e.g. LFW-style benchmarks) to get real TAR/FAR numbers, not vendor claims.
- **ANPR data:** Indian plate image dataset (public sets exist — search for "Indian number plate dataset" on Kaggle/GitHub) plus your own photos of demo vehicles at the expected capture angle/distance, since `PLATE_TEXT_NORM`'s format-aware logic needs real Indian-format examples to validate against.
- **Watchlist / Authorized DB seed data:** synthetic identities and plates only — never real watchlist data. Design the schema now (`category`, `validity_period`, `source_authority` per the `ENROLL` node) so `AUTH_DB` doesn't need a schema migration later.
- **Evaluation dataset split:** carve out a held-out slice of everything above *before* any model tuning happens — this becomes `EVAL_DATASET` and must stay untouched until Phase 8. Label it (`GROUND_TRUTH`) with true class / true identity match / true plate string.
- Build the RTSP/ONVIF ingestion service and a `CAM_CONFIG` schema (ROI, capture zone, angle, illumination, per-camera quality thresholds).

**Exit criteria:** ingestion pipeline pulls a live or looped RTSP stream into the pipeline; datasets are versioned (DVC or even just a dated folder + checksum) and the eval split is locked and untouched.

---

## Phase 2 — Core AI/ML Pipelines (built and validated standalone)

**Implements:** `DET`→`TRACK`→`RULE_EVAL`→`FA_FILTER`→`PERIMETER_RESULT`; `FACE_DET`→`LIVE`→`FACE_EMB`→`FACE_MATCH`→`FACE_RESULT`; `PLATE_DET`→`OCR`→`PLATE_TEXT_NORM`→`PLATE_VALID`→`PLATE_MATCH`→`PLATE_RESULT`; `MANUAL_REVIEW`

Build each pipeline **independently and get it working in isolation** before Phase 3 tries to fuse them — this is the single biggest time-saver for a hackathon build.

Perimeter:
- YOLO26-N object detection, ByteTrack for multi-object tracking, zone/line-crossing rule evaluation, the false-alarm filter (a lightweight learned or heuristic background model to kill vegetation/animal triggers).
- Output must conform to the `PERIMETER_RESULT` schema (class, track, zone-event, confidence).

Face:
- SCRFD detection → Quality/Liveness gate — **build the honest MVP version only**: a face-quality gate (blur, pose, illumination checks) and controlled capture. Do not build or claim a Presentation Attack Detection model unless you genuinely have time; the v5 diagram already commits you to being honest about this distinction, so don't let implementation quietly drift into overclaiming.
- ArcFace embeddings (InsightFace buffalo_l) → 1:1 document match / 1:N watchlist search → `FACE_RESULT` with explicit status (`authorized` / `watchlist-match` / `unknown` / `manual`).
- Recapture loop on quality/liveness failure (max 2 attempts) → `MANUAL_REVIEW`.

ANPR:
- YOLO26-N fine-tuned for Indian plates → PaddleOCR → the format-aware, position-specific normalization step (never a blind global character replace — validate this against your real Indian-plate dataset from Phase 1) → format validation → `PLATE_RESULT`.

Manual Review Queue:
- One unified queue/UI stub for: low-quality face, low-confidence match, low-confidence OCR, partial plate, retries exhausted. Doesn't need to be pretty yet — it needs to exist as a real endpoint, because Phase 3's ambiguous person-vehicle cases route here too.

**Exit criteria:** each of the three pipelines runs end-to-end on Phase 1 data and produces output conforming exactly to its `*_RESULT` schema. Preliminary accuracy/latency numbers recorded (final numbers come in Phase 8, but you want an early read to catch a broken pipeline before Phase 3 depends on it).

---

## Phase 3 — Event Fusion & Decision Engine

**This is the architectural core of the project and the hardest phase — budget the most time here.**

**Implements:** `EVENT_TRIGGER` → `EVENT_KEY` / `FUSION_WINDOW` (open-window registry) → `SIGNAL_ASSOC` → `ORPHAN_SIGNAL` → `EVENT_FUSION` → `PV_LINK` → `PV_CANDIDATE_GEN`/`PV_CANDIDATE_SCORE`/`PV_UNIQUE` → `PV_CORRELATED`/`PV_AMBIGUOUS`/`PV_VEHICLE_ONLY`/`PV_FACE_ONLY` → `EVENT_FINALIZE` → `FACE_COMP`/`PLATE_COMP`/`ZONE_COMP`/`BEHAVIOR_COMP` → `*_SCORE_NORM` → `SIGNAL_AVAIL`/`WEIGHT_RENORM` → `RISK_ENGINE` → `RISK_BAND` → `RISK_DECISION`; `POLICY_RULES` → `POLICY_DECISION`; `FINAL_ALERT` → `DECISION_EXPLAIN` / `LOG_EVENT`

Build in this exact sub-order — each step is a hard dependency of the next:

1. **Event Trigger + Open-Window Registry.** BOP: perimeter result confirms a zone/line-crossing. BCP: `ANPR_CAM`'s physical loop/beam sensor fires (not the Jetson — get this source right, it's a distinction a judge will probe). Each trigger creates `event_id = UUID + start_time + camera_id + lane_id` and opens a bounded-duration window in the registry.
2. **Signal-to-Event Association.** Every incoming `PERIMETER_RESULT` / `FACE_RESULT` / `PLATE_RESULT` arrives carrying only `camera_id + lane_id + timestamp` — it does **not** know its own `event_id`. This step looks it up against the open-window registry and assigns it. Build the unmatched-signal path (`ORPHAN_SIGNAL` → audit log) at the same time — don't leave it for later, it's a two-line branch now and a re-architecture later.
3. **De-duplication + Person-Vehicle Correlation.** Merge repeat detections within the window, then run the candidate-generation → candidate-scoring → unique-candidate? sequence for BCP events. Explicitly test the four outcomes: correlated, ambiguous (route to `MANUAL_REVIEW`), vehicle-only, person-only.
4. **Component scoring with conditional weight renormalization.** This is the part most teams skip and it's a P0 architectural requirement: a BCP event only ever has face+plate data (zone/behavior are absent, not zero), a perimeter event only ever has zone+behavior. Build the "which signals are actually available" check and the weight-renormalization math (base weights 0.4/0.3/0.2/0.1, rescaled over whatever's present) before writing the risk formula — if you hardcode the 4-weight formula first you will have to redo it.
5. **Risk Decision and Policy Decision, built as genuinely separate code paths**, both feeding one `FINAL_ALERT` via OR. Test explicitly that a watchlist match with a *low* numeric risk score still alerts.
6. **Decision + Explanation object.** Six distinct fields, never merged into one string: detection, watchlist match, PV correlation, risk assessment, policy violation, human verification. This object is what your dashboard and your pitch deck's "explainability" claim both depend on — get the schema right once.

**Exit criteria:** feed the engine a scripted sequence of synthetic events (single correlated BCP event, ambiguous multi-face/multi-vehicle BCP event, vehicle-only event, perimeter-only event, a signal arriving after its window has closed, a watchlist match with an otherwise-low risk score) and verify each produces the architecturally-correct outcome. Write these as actual automated tests — they're also your Phase 7 regression suite.

---

## Phase 4 — Backend, Data & Security Layer

**Implements:** `EVENT_API`, `ALERT_SVC`, `WATCHLIST_SVC`, `CLIP_SVC`, `SYNC_SVC`; `EVENT_DB`, `EVIDENCE_STORE`, `AUTH_DB`, `MODEL_REG`, `AUDIT_LOG`; `TLS`/`ENC`/`RBAC`/`HASHCHAIN`

Tasks:
- FastAPI event ingestion endpoint conforming to the fusion engine's output schema.
- Alert generation service, wired to trigger evidence clip extraction (`CLIP_SVC`: pre+post-buffer clip → SHA-256 hash → keyed to `event_id`) only on alert or a selected checkpoint event — not on every routine crossing, or your evidence store fills up fast.
- Postgres + TimescaleDB for `EVENT_DB`; MinIO/S3 (or local equivalent for the demo) encrypted at rest for `EVIDENCE_STORE`.
- Hash-chained, tamper-evident `AUDIT_LOG` — every privileged access and every decision gets an entry, including the `ORPHAN_SIGNAL` path from Phase 3.
- Store-and-forward sync (`SYNC_SVC`) so an edge device buffers locally if the network drops — this doubles as your live-demo safety net (see Phase 9).
- RBAC: implement the five roles as they're defined in the diagram (Operator, Checkpoint Officer, Supervisor, Administrator, Auditor) with face images/evidence clips scoped separately from general event data — don't grant blanket access "for now and fix it later," because retrofitting RBAC after the dashboard is built is much more painful than building it in from the start.
- TLS/mTLS on all edge-to-backend and internal service calls.

**Exit criteria:** an event posted to the API is persisted, an alert triggers a real evidence clip with a verifiable hash, RBAC actually blocks a Checkpoint Officer from seeing another checkpoint's events, and the audit log entry exists and is tamper-evident (test by trying to edit an entry directly in the DB and confirming the hash chain breaks visibly).

---

## Phase 5 — Command & Control Dashboards

**Implements:** `AUTHN`, `DASH_BOP`, `DASH_BCP`, `DASH_CENTRAL`, `HV_PERI`/`HV_BCP`, `RESPONSE`

Tasks:
- OAuth2/OIDC login wired to the Phase 4 RBAC roles.
- BOP/Sector Operator dashboard: live perimeter alerts, confirm/dismiss.
- Checkpoint Officer console: live checkpoint alerts with the full `DECISION_EXPLAIN` object rendered legibly (this is your explainability demo moment — don't just show a risk number, show the six separated claim types) and clear/escalate actions.
- Central MHA rollup dashboard: simulate 2–3 BOPs feeding in, as scoped in the diagram — don't try to build a real multi-site system for the hackathon.
- Response/action logging (QRT dispatch / stop-clear-escalate) tied back to the audit log.

**Exit criteria:** an operator can see an alert land in near-real-time from a test event, open the explanation object, and confirm/dismiss/escalate — and that action is both role-scoped and audit-logged.

---

## Phase 6 — Governance, Resilience & KPI/Evaluation Layer

**Implements:** `CAM_CONFIG` (already started in Phase 1, finish here), `ENROLL`, `THRESH` (with `EVAL_DATASET`→`GROUND_TRUTH`→`METRIC_CALC` wired in), `SYS_HEALTH`, `MODEL_MONITOR`, `RETENTION`; `GPU_INFER`→`CPU_FALLBACK`→`PERF_WARN`; `CAM_OFFLINE`→`NOTIFY_OP`→`NO_FALSE_CLEAR`; `KPI_PERI`/`KPI_FACE`/`KPI_ANPR`/`KPI_SYS`, `LAT_BUDGET`

Tasks:
- **Threshold calibration, done for real:** run the ROC/TAR-FAR curve on `EVAL_DATASET`, pick `T_low`/`T_high`/`T_critical` from that curve, then confirm on the held-out test split. Do not hardcode illustrative numbers from the architecture doc into production — those were placeholders, not your actual thresholds.
- Identity enrollment workflow UI/flow (source agency submits → authorized official verifies → record added).
- System health monitor with the explicit failure paths: GPU failure → CPU fallback → **visible** health warning (never silently reports normal performance); camera disconnect → marked offline → operator notified → **no auto-generated alert from missing data** (this "no detection ≠ no activity" rule is a genuine safety property — test it directly by unplugging a camera and confirming no false "all clear" state appears anywhere).
- Retention policy: implement actual configurable durations (not the illustrative N/M/K from the diagram — pick real values appropriate for your demo and document them as config, not hardcoded constants) plus a working automated expiry job you can demonstrate.
- Wire up the KPI dashboards fed by `METRIC_CALC`, and set real latency budget numbers once you've benchmarked (Phase 7/8), not guessed ones.

**Exit criteria:** you can pull the GPU offline mid-demo and watch the system degrade visibly and honestly instead of silently; you can disconnect a camera and confirm no phantom "clear" state; your calibrated thresholds have a documented ROC curve behind them, not a guess.

---

## Phase 7 — Integration & System Testing

Tasks:
- **Unit → component → integration → end-to-end** test pyramid. Phase 3's scripted event sequences become your integration regression suite.
- Specific end-to-end scenarios to script and log results for:
  1. BOP perimeter intrusion → alert → operator confirms → evidence clip verified.
  2. BCP single face + single vehicle → correlated → risk+policy evaluated → dashboard shows full explanation.
  3. BCP ambiguous multi-face/multi-vehicle → routed to manual review → officer resolves → event finalized.
  4. Watchlist match at low numeric risk → still alerts (Policy Decision override) — this is the test that proves your P0 architectural fix actually works in code, not just on paper.
  5. Signal arriving after its fusion window has closed → lands in `ORPHAN_SIGNAL`, audit-logged, not silently dropped.
  6. Camera goes offline mid-session → no false "clear," operator notified.
  7. GPU failure → CPU fallback → visible degraded-performance warning.
- Latency measurement against the `LAT_BUDGET` placeholders (detection, face recognition, ANPR, fusion, alert generation, end-to-end) — record actual numbers.
- Security pass: RBAC boundary tests (can a Checkpoint Officer see BOP-only data? should be no), audit log tamper-evidence check.

**Exit criteria:** all seven scenarios pass and are logged with evidence (screenshots/logs), latency numbers are real and documented, no RBAC boundary failures.

---

## Phase 8 — Evaluation Against KPIs

Tasks:
- Run `METRIC_CALC` against `EVAL_DATASET`/`GROUND_TRUTH` for real precision/recall/FAR/TAR/accuracy per component (perimeter, face, ANPR) — this is the number set you'll defend in front of judges.
- Compare against the SIH26187 problem statement's stated requirements (if MHA specified target accuracy/latency figures, check them explicitly).
- Write up the KPI scorecard in "measured, not claimed" language — state the eval dataset size and composition alongside every number, since a judge may ask how you got it.

**Exit criteria:** a one-page KPI scorecard exists with real numbers, dataset provenance, and no unsupported claims.

---

## Phase 9 — Deployment & Demo Environment Setup

Tasks:
- Package the edge appliance build (Jetson image + the CPU-fallback config path) so it's reproducible, not hand-configured on one laptop.
- Set up the physical or simulated demo rig: at minimum one BOP feed and one BCP lane, ideally matching whatever hardware SIH provides at your venue.
- Seed the demo environment with dedicated demo identities/plates in `AUTH_DB` (never real watchlist data) and pre-stage the scenario clips from Phase 7 so you can trigger them reliably on stage.
- Rehearse the offline/degraded-mode fallback specifically for the live demo — if venue wifi drops, you want `SYNC_SVC`'s buffering to save you, and you want a recorded backup video as a last resort.

**Exit criteria:** you can tear down and stand the whole demo rig back up from scratch in under the time you'll actually have at the venue, at least twice, without a team member manually patching something.

---

## Phase 10 — Final Demonstration & Judge Q&A Prep

Tasks:
- Build a live demo script around 4–5 flagship moments: a correlated BCP event with full explanation shown, an ambiguous case correctly routed to manual review, a camera-offline/GPU-fallback moment shown live (this is a strong differentiator — most teams hide failure modes, you should show yours handling one gracefully), and a watchlist-match-at-low-risk alert to demonstrate the Policy Decision path.
- Rehearse the specific judge question already flagged as high-priority: **"no facial recognition on the perimeter vs. watchlist matching at checkpoints."** Have a crisp one-line answer ready (perimeter never runs face recognition at all — it's zone/behavior only; face recognition exists exclusively at the check-post, gated by consent/authorization workflow, doing 1:1 document match or 1:N watchlist search, never open-ended surveillance).
- Prep answers for the architecture questions this project has already been pressure-tested on: how `event_id` gets assigned to a signal that arrives before the event exists (point to `SIGNAL_ASSOC`), why missing signals aren't treated as zero (point to `SIGNAL_AVAIL`/`WEIGHT_RENORM`), and why liveness is a quality gate at MVP rather than a claimed PAD model (say so plainly — don't let a judge catch an overclaim).
- Align the PPT wording with what's actually built vs. Phase 2/3 — every slide claim should be traceable to a diagram node's classDef (solid = built, dashed = future).

**Exit criteria:** full dry run in front of someone outside the team, timed, with the Q&A cheat-sheet used cold to check it actually holds up under questioning.

---

## If your runway is short: what to cut first

In rough order of "safest to defer without breaking the architecture's integrity":

1. Central rollup dashboard simulating multiple BOPs → demo with a single BOP + single BCP instead.
2. Model drift monitoring (`MODEL_MONITOR`) → note it as designed-but-not-instrumented for MVP.
3. Full RBAC role set → implement Operator + Officer + Admin only, note Supervisor/Auditor as designed.
4. Automated retention expiry job → implement the policy and schema, demonstrate manual expiry trigger instead of a scheduler.

**Do not cut**, even under time pressure — these are exactly the parts your architecture reviews identified as the load-bearing differentiators:
- The Signal-to-Event Association mechanism (Phase 3, step 2) — this is the answer to the hardest judge question.
- Conditional weight renormalization (Phase 3, step 4) — without it the risk engine is mathematically broken for partial events, which is likely to come up.
- Risk Decision / Policy Decision separation (Phase 3, step 5).
- The camera-offline / GPU-fallback honesty paths (Phase 6) — these are cheap to build and are a strong live-demo moment precisely because most teams don't have them.

---

## Definition of Done — quick reference table

| Phase | Done when |
|---|---|
| 0 | Scope frozen in writing, repo + CI live |Confirmed. **Step 4.8K is now ready for implementation.**

The domain model and PostgreSQL schema align cleanly:

* `CameraDomain` ↔ `TEXT`
* `frozenset[str]` ↔ `TEXT[]`
* repository remains persistence-only
* no authorization logic inside the repository
* `CameraScopeRegistry` remains the domain-level lookup mechanism

### Step 4.8K — Create repository + integration tests

Run exactly:

```bash
cd /Users/hariharans/Documents/SIH26187
source .venv/bin/activate

cat > backend/camera_scope_repository.py <<'PY'
from __future__ import annotations

import psycopg

from backend.camera_scope import CameraDomain, CameraScope, CameraScopeRegistry


class PostgresCameraScopeRepository:
    def __init__(self, dsn: str):
        self._dsn = dsn

    def upsert(self, scope: CameraScope) -> None:
        sql = """
        INSERT INTO camera_scopes (
            camera_id,
            domain,
            bop_id,
            checkpoint_id,
            lane_ids
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (camera_id)
        DO UPDATE SET
            domain = EXCLUDED.domain,
            bop_id = EXCLUDED.bop_id,
            checkpoint_id = EXCLUDED.checkpoint_id,
            lane_ids = EXCLUDED.lane_ids
        """

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        scope.camera_id,
                        scope.domain.value,
                        scope.bop_id,
                        scope.checkpoint_id,
                        sorted(scope.lane_ids),
                    ),
                )

    def get(self, camera_id: str) -> CameraScope | None:
        sql = """
        SELECT
            camera_id,
            domain,
            bop_id,
            checkpoint_id,
            lane_ids
        FROM camera_scopes
        WHERE camera_id = %s
        """

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (camera_id,))
                row = cur.fetchone()

        if row is None:
            return None

        return self._row_to_scope(row)

    def list_all(self) -> list[CameraScope]:
        sql = """
        SELECT
            camera_id,
            domain,
            bop_id,
            checkpoint_id,
            lane_ids
        FROM camera_scopes
        ORDER BY camera_id
        """

        with psycopg.connect(self._dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()

        return [self._row_to_scope(row) for row in rows]

    def to_registry(self) -> CameraScopeRegistry:
        return CameraScopeRegistry(self.list_all())

    @staticmethod
    def _row_to_scope(row: tuple) -> CameraScope:
        camera_id, domain, bop_id, checkpoint_id, lane_ids = row

        return CameraScope(
            camera_id=camera_id,
            domain=CameraDomain(domain),
            bop_id=bop_id,
            checkpoint_id=checkpoint_id,
            lane_ids=frozenset(lane_ids or []),
        )
PY

cat > tests/test_camera_scope_repository.py <<'PY'
import uuid

import psycopg
import pytest

from backend.camera_scope import CameraDomain, CameraScope
from backend.camera_scope_repository import PostgresCameraScopeRepository


DSN = "dbname=sih26187"


def unique_camera_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def repository():
    repository = PostgresCameraScopeRepository(DSN)

    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM camera_scopes")

    yield repository

    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM camera_scopes")


def test_upsert_and_get_bop_scope(repository):
    scope = CameraScope(
        camera_id=unique_camera_id("cam_bop"),
        domain=CameraDomain.BOP,
        bop_id="BOP-01",
    )

    repository.upsert(scope)

    result = repository.get(scope.camera_id)

    assert result == scope


def test_upsert_and_get_bcp_scope_with_lanes(repository):
    scope = CameraScope(
        camera_id=unique_camera_id("cam_bcp"),
        domain=CameraDomain.BCP,
        checkpoint_id="BCP-01",
        lane_ids=frozenset({"LANE-02", "LANE-01"}),
    )

    repository.upsert(scope)

    result = repository.get(scope.camera_id)

    assert result == scope
    assert result.lane_ids == frozenset({"LANE-01", "LANE-02"})


def test_upsert_replaces_existing_scope(repository):
    camera_id = unique_camera_id("cam_update")

    original = CameraScope(
        camera_id=camera_id,
        domain=CameraDomain.BOP,
        bop_id="BOP-01",
    )

    updated = CameraScope(
        camera_id=camera_id,
        domain=CameraDomain.BCP,
        checkpoint_id="BCP-01",
        lane_ids=frozenset({"LANE-01"}),
    )

    repository.upsert(original)
    repository.upsert(updated)

    assert repository.get(camera_id) == updated


def test_get_unknown_camera_returns_none(repository):
    assert repository.get(unique_camera_id("missing")) is None


def test_list_all_returns_scopes_in_camera_id_order(repository):
    scope_b = CameraScope(
        camera_id=unique_camera_id("cam_b"),
        domain=CameraDomain.BOP,
        bop_id="BOP-01",
    )

    scope_a = CameraScope(
        camera_id=unique_camera_id("cam_a"),
        domain=CameraDomain.BCP,
        checkpoint_id="BCP-01",
    )

    repository.upsert(scope_b)
    repository.upsert(scope_a)

    result = repository.list_all()

    assert [scope.camera_id for scope in result] == sorted(
        [scope_a.camera_id, scope_b.camera_id]
    )


def test_to_registry_contains_persisted_scopes(repository):
    scope = CameraScope(
        camera_id=unique_camera_id("cam_registry"),
        domain=CameraDomain.BCP,
        checkpoint_id="BCP-01",
        lane_ids=frozenset({"LANE-01"}),
    )

    repository.upsert(scope)

    registry = repository.to_registry()

    assert len(registry) == 1
    assert registry.resolve(scope.camera_id) == scope


@pytest.mark.parametrize(
    "values",
    [
        (
            "BOP",
            None,
            "BCP-01",
            [],
        ),
        (
            "BOP",
            "BOP-01",
            None,
            ["LANE-01"],
        ),
        (
            "BCP",
            "BOP-01",
            "BCP-01",
            [],
        ),
        (
            "BCP",
            None,
            None,
            [],
        ),
    ],
)
def test_database_rejects_invalid_scope_states(repository, values):
    camera_id = unique_camera_id("cam_invalid")
    domain, bop_id, checkpoint_id, lane_ids = values

    with pytest.raises(psycopg.errors.CheckViolation):
        with psycopg.connect(DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO camera_scopes (
                        camera_id,
                        domain,
                        bop_id,
                        checkpoint_id,
                        lane_ids
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        camera_id,
                        domain,
                        bop_id,
                        checkpoint_id,
                        lane_ids,
                    ),
                )
PY

python3 -m py_compile backend/camera_scope_repository.py tests/test_camera_scope_repository.py

PYTHONPATH=.:.venv/lib/python3.10/site-packages pytest -q \
    tests/test_camera_scope_repository.py \
    tests/test_rbac_contract.py \
    tests/test_rbac_scope_integration.py \
    tests/test_event_api_rbac.py
```

### Expected scope of this step

This should prove:

```text
Repository
├── BOP round-trip                    ✅
├── BCP + lanes round-trip            ✅
├── update/upsert                     ✅
├── missing camera                    ✅
├── deterministic list                ✅
├── registry conversion               ✅
└── PostgreSQL invariant enforcement  ✅
```

And importantly, **we are not connecting this repository to `backend/app.py` yet**. That integration would change the runtime RBAC path and deserves its own controlled step.

Run the command and send me the complete pytest output.

| 1 | Datasets versioned, eval split locked and untouched |
| 2 | All three pipelines run standalone, output matches `*_RESULT` schemas |
| 3 | All 6 scripted event scenarios produce architecturally-correct outcomes |
| 4 | RBAC blocks cross-checkpoint access, audit log tamper-evident |
| 5 | Operator can view, confirm/dismiss/escalate, action is logged |
| 6 | GPU/camera failure demoed live without a false "clear" state |
| 7 | All 7 end-to-end scenarios pass with logged evidence |
| 8 | KPI scorecard has real, sourced numbers |
| 9 | Demo rig rebuilds from scratch twice, reliably |
| 10 | Dry run completed cold in front of an outsider |
