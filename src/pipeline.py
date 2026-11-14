"""YOUR PoC PIPELINE — this is the file your group owns.

Lab contract = 5 automated steps, run by `python -m src.main ...`:
    01 load      input + provenance check (manifest.csv)
    02 process   your algorithm / model (the research part)
    03 rank      prioritise for a fixed review budget B
    04 review    human-in-the-loop: queue -> reviewer decisions -> overrides + audit log
    05 decide    GO / HOLD / REWORK (or track equivalent) with reason_code + owner, write outputs

Required output: out/<pred_file> with the columns listed for your track in
 docs/tracks/TXX.md (also in src/metrics/tracks.py::TRACKS). Then evaluate with
    python -m src.evaluate --track tXX --pred out/<pred_file>

Rules: never read fixtures/*/gt/ here (the GT guard will stop you); seed everything via ctx.rng;
no network calls; fail loudly on missing models instead of downloading.
See src/examples/random_baseline.py for a minimal end-to-end wiring example.
"""
from __future__ import annotations

import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import apply_overrides, load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict:
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    data = {"manifest": pd.read_csv(manifest) if manifest.exists() else None}
    images = ctx.input_dir / "images.csv"
    if images.exists():
        data["images"] = pd.read_csv(images)
    return data


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    """Minimal but reviewable T02 implementation.

    The goal here is not to replace the full research detector yet; it is to keep the pipeline
    operational so the human review loop, audits, and decision record can be exercised end to end.
    """
    spec = get_track(ctx.track)
    images = data.get("images")
    if images is None:
        raise ValueError(f"track {ctx.track} requires images.csv under {ctx.input_dir}")
    if "image_id" not in images.columns:
        raise ValueError("images.csv must include image_id")

    out = images[["image_id"]].copy()
    out["path"] = images["path"].astype(str) if "path" in images.columns else ""

    # Deterministic triage signal: night/dusk and camera_B are more suspicious in this fixture.
    light = images["light"].astype(str) if "light" in images.columns else pd.Series(["day"] * len(images), index=images.index)
    camera = images["camera_id"].astype(str) if "camera_id" in images.columns else pd.Series(["cam_A"] * len(images), index=images.index)

    light_score = {"night": 1.0, "dusk": 0.6, "day": 0.2}
    camera_score = {"cam_B": 0.5, "cam_A": 0.1}
    out["score"] = light.map(light_score).fillna(0.0).to_numpy() + camera.map(camera_score).fillna(0.0).to_numpy()
    out["reason"] = "triage:" + light.fillna("unknown").astype(str) + ";" + camera.fillna("unknown").astype(str)

    # Keep the output shape expected by T02 and by the review queue helper.
    return out[["image_id", "path", "score", "reason"]]


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    """TI02 review queue: top-B by score, then render offline contact sheet."""
    budget = int(ctx.param("budget", 20))
    if "score" not in scored.columns:
        raise ValueError("rank(): process() output needs a numeric `score` column")
    if "image_id" not in scored.columns:
        raise ValueError("rank(): process() output needs an `image_id` column")

    q = make_review_queue(
        scored,
        id_col="image_id",
        score_col="score",
        budget=budget,
        reason_col="reason" if "reason" in scored.columns else None,
        evidence_col="path" if "path" in scored.columns else None,
    )
    write_review_queue(q, ctx.out("review_queue.csv"))

    render_contact_sheet(
        q,
        ctx.out("contact_sheet.html"),
        image_col="evidence_path",
        image_root=ctx.input_dir,
        title=f"{ctx.track.upper()} review queue",
        fields=("rank", "item_id", "score", "reason"),
    )
    return q


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """If a reviewed queue exists (out/review_queue_reviewed.csv), apply it. Otherwise pass through.

    The human-review CSV is produced by the HTML contact sheet and contains columns including
    `item_id`, `reviewer`, `reviewer_decision`, `override_value`, and `review_note`.
    """
    final = scored.copy()
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if reviewed_path.exists():
        reviewed = load_review_decisions(reviewed_path)
        ctx.stats["n_reviewed"] = len(reviewed)

        review_cols = ["image_id", "reviewer", "reviewer_decision", "override_value", "review_note"]
        merged = final[["image_id"]].merge(
            reviewed.rename(columns={"item_id": "image_id"})[review_cols],
            on="image_id",
            how="left",
        )
        final = final.merge(merged, on="image_id", how="left")

        final = apply_overrides(final, reviewed, id_col="image_id", value_col="score", audit=ctx.audit)

        if ctx.audit:
            for row in reviewed.itertuples(index=False):
                ctx.audit.log(
                    actor=row.reviewer,
                    action=f"review:{row.reviewer_decision}",
                    item_id=row.item_id,
                    before=None,
                    after=row.override_value or row.reviewer_decision,
                    reason=row.review_note,
                )
    return final


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    """T02 decision policy: KEEP if the review queue remains acceptable, RECAPTURE when reviewers or
    risk signals push the batch back for correction.
    """
    owner = str(ctx.config.get("owner", "TODO-owner")).strip() or "TODO-owner"

    decisions = pd.Series([], dtype=str)
    if "reviewer_decision" in final.columns:
        decisions = final["reviewer_decision"].fillna("").astype(str).str.strip().str.lower()

    human_recap = decisions.isin({"reject", "escalate"}).any() if len(decisions) else False
    score_col = "score" if "score" in final.columns else None
    high_risk = False
    if score_col is not None:
        score = final[score_col].fillna(0.0)
        high_risk = bool((score >= float(ctx.param("re_capture_threshold", 0.9))).any())

    if human_recap or high_risk:
        status = "RECAPTURE"
        reason_code = "REVIEW_RECAPTURE"
        reason = "Human review or score threshold indicates a risky queue that should be recaptured before release."
    else:
        status = "KEEP"
        reason_code = "LOW_RISK_KEEP"
        reason = "Queue remains within acceptable risk; no override required for release."

    evidence = [str(ctx.out("review_queue.csv")), str(ctx.out("contact_sheet.html"))]
    if ctx.audit is not None:
        evidence.append(str(ctx.audit.path))
    evidence.append(str(ctx.out("decision.json")))

    return Decision(
        status=status,
        reason_code=reason_code,
        reason=reason,
        owner=owner,
        scope=ctx.track,
        evidence=evidence,
    )


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    final[spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
