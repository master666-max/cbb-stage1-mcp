"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> int:
    from cbb_stage1.config import Config
    from cbb_stage1.orchestrator.pipeline import run_pipeline

    cfg_kwargs = {
        "book_id": args.book_id,
        "book_path": Path(args.book),
        "out_dir": Path(args.out),
        "work_dir": Path(args.work),
        "live": args.live,
    }
    if args.provider:
        cfg_kwargs["provider"] = args.provider
    if args.model:
        cfg_kwargs["model"] = args.model
    if args.nli_provider:
        cfg_kwargs["nli_provider"] = args.nli_provider
    if args.nli_model:
        cfg_kwargs["nli_model"] = args.nli_model
    if getattr(args, "narrator_reliability", None):
        # D2: declaring the narrator unreliable downgrades narration-only
        # assertions to the statement layer instead of asserting them as facts.
        cfg_kwargs["narrator_reliability"] = args.narrator_reliability
    cfg = Config.load(args.config) if args.config else Config(**cfg_kwargs)
    result = run_pipeline(cfg, chapters=args.chapters)
    print(f"status={result.status} units={result.units_processed}/{result.units_total} work={result.work_dir}")
    if args.box:
        from cbb_stage1.box.writer import build_box

        build_box(
            result.work_dir,
            args.out,
            series_id=args.series_id,
            volume_no=args.volume_no,
        )
        print(f"box={args.out}")
    return 0


def cmd_build_box(args: argparse.Namespace) -> int:
    from cbb_stage1.box.writer import build_box

    build_box(
        args.work,
        args.out,
        series_id=args.series_id,
        volume_no=args.volume_no,
    )
    print(f"box={args.out}")
    return 0


def cmd_lint_box(args: argparse.Namespace) -> int:
    from cbb_stage1.box.lint import lint_box

    problems = lint_box(Path(args.box))
    for p in problems:
        print(p)
    return 0 if not problems else 1


def cmd_acceptance(args: argparse.Namespace) -> int:
    from cbb_stage1.qc.acceptance import verify

    return 0 if verify(Path(args.matrix), run=args.run) else 1


def cmd_wiring(args: argparse.Namespace) -> int:
    from cbb_stage1.qc.wiring import main as wiring_main

    return wiring_main(["--check"] if args.check else [])


def cmd_model_policy(args: argparse.Namespace) -> int:
    from cbb_stage1.model_ops import write_policy

    result = write_policy(
        args.work,
        provider=args.provider,
        model=args.model,
        transport=args.transport,
        host_queue=args.host_queue,
        mcp_command=args.mcp_command,
        mcp_tool=args.mcp_tool,
        embedding_base_url=args.embedding_base_url,
        embedding_model=args.embedding_model,
        max_concurrency=args.max_concurrency,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        token_budget=args.token_budget,
        role_models=args.role_models,
        session_model_label=args.session_model_label,
        judge_base_url=args.judge_base_url,
        judge_model=args.judge_model,
        nli_base_url=args.nli_base_url,
        nli_model=args.nli_model,
        whitebox_backend=args.whitebox_backend,
    )
    print(f"policy={result['data']['policy']}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Pre-flight dependency audit: probe first, then decide.

    Exists because nothing used to check dependencies before a run: a dead
    embedding endpoint or an already-existing box directory only surfaced tens
    of minutes into the pipeline, at the stage that needed them.

    Every value a probe needs is read from the same sources the run itself uses
    (CLI flag → environment → ``Config`` default), so the audit cannot report a
    dependency READY while the run would see it as absent.
    """
    import json
    import os

    from cbb_stage1.config import Config
    from cbb_stage1.preflight import format_report, run_preflight

    def pick(flag: object, env_name: str, config_attr: str, default: object = None) -> object:
        if flag is not None:
            return flag
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            try:
                return int(env_value)
            except ValueError:
                return env_value
        return getattr(Config, config_attr, None) if default is None else default

    # ``Config`` carries dataclass defaults; read them without instantiating a
    # full config (that would require a book path we may not have yet).
    def config_default(name: str) -> object:
        field = Config.__dataclass_fields__.get(name) if hasattr(Config, "__dataclass_fields__") else None
        return field.default if field is not None else None

    skip = {item.strip() for item in (args.skip or "").split(",") if item.strip()}
    report = run_preflight(
        book=args.book,
        work=args.work,
        out=args.out,
        embedding_base_url=args.embedding_base_url,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        # Who is serving the main reasoning. Under the agent-driven architecture
        # this varies per run, so the capability set is not a project constant —
        # the probe records the transport-determined part and marks the rest as
        # depending on the driver.
        transport=getattr(args, "transport", None),
        session_model_label=getattr(args, "session_model_label", None)
        or os.environ.get("CBB_SESSION_MODEL_LABEL"),
        base_url=args.llm_base_url,
        nli_config={
            "nli_provider": args.nli_provider or config_default("nli_provider"),
            "nli_model": args.nli_model or config_default("nli_model"),
        },
        uq_head_checkpoint=args.uq_head_checkpoint
        or os.environ.get("CBB_UQ_HEAD_CHECKPOINT")
        or config_default("uq_head_checkpoint"),
        uq_train_data=args.uq_train_data or os.environ.get("CBB_UQ_TRAIN_DATA"),
        max_tokens=pick(args.max_tokens, "CBB_LLM_MAX_TOKENS", "max_tokens"),
        token_budget=pick(args.token_budget, "CBB_LLM_TOKEN_BUDGET", "token_budget"),
        max_workers=pick(args.max_workers, "CBB_MAX_WORKERS", "max_workers"),
        skip=skip,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    return 0 if report["status"] == "READY" else 1


def cmd_lexicon_list(args: argparse.Namespace) -> int:
    """D5: show unregistered-predicate proposals and their status."""
    import json

    from cbb_stage1.assemble.lexicon_evolution import list_proposals

    rows = list_proposals(args.work)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("no proposals")
        return 0
    print(f"{'predicate':16s} {'status':10s} {'count':>6s}  seen_in")
    for row in rows:
        seen = ", ".join(str(u) for u in (row.get("seen_in") or [])[:3]) or "-"
        print(
            f"{str(row.get('predicate') or ''):16s} "
            f"{str(row.get('status') or ''):10s} "
            f"{int(row.get('count') or 0):6d}  {seen}"
        )
    return 0


def cmd_lexicon_approve(args: argparse.Namespace) -> int:
    """D5: approve one proposal -> register it, then report units to re-scan.

    Refuses without ``--by``: I4 says predicates are not grown from data, so the
    human gate has to be explicit rather than defaulted.
    """
    import json

    from cbb_stage1.assemble.lexicon_evolution import ApprovalError, approve_proposal

    try:
        result = approve_proposal(
            args.work,
            args.predicate,
            approved_by=args.by or "",
            library=args.library,
            lexicon_path=args.lexicon,
        )
    except ApprovalError as exc:
        print(f"refused: {exc}")
        return 2
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"approved {result.predicate!r} in library {result.library!r}")
        print(f"  by     : {result.approved_by}")
        print(f"  at     : {result.approved_at}")
        print(f"  lexicon: {result.lexicon_path}")
        print(f"  rescan : {', '.join(result.units_to_rescan) or '(none recorded)'}")
        print(f"  note   : {result.note}")
    return 0


def cmd_supersede(args: argparse.Namespace) -> int:
    """D9: mark one assertion superseded by another. Human decision, not automatic.

    ``apply_supersede`` was written and never reachable. It is deliberately NOT
    wired into ``validate``: superseding a record is an editorial judgement
    (``04-执行模型`` §8: "decisions 本轮不自动裁决"), so it needs a command a person
    runs, not a step the pipeline performs.
    """
    import json

    from cbb_stage1.validate.conflicts import apply_supersede

    work = Path(args.work)
    table = work / "assemble" / "assertions.jsonl"
    if not table.exists():
        print(f"no assertion table at {table}")
        return 2
    records = [
        json.loads(line)
        for line in table.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    try:
        plan = apply_supersede(
            records,
            loser_id=args.loser,
            winner_id=args.winner,
            decided_by=args.by,
            reason=args.reason,
            at=args.at,
        )
    except ValueError as exc:
        print(f"refused: {exc}")
        return 2

    table.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in plan.records) + "\n",
        encoding="utf-8",
    )
    decisions = work / "audit" / "decisions.jsonl"
    decisions.parent.mkdir(parents=True, exist_ok=True)
    with decisions.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(plan.decision, ensure_ascii=False) + "\n")

    if args.json:
        print(json.dumps(
            {"decision": plan.decision, "retcon": plan.retcon},
            ensure_ascii=False, indent=2,
        ))
    else:
        print(f"superseded {args.loser} -> {args.winner}")
        print(f"  by    : {args.by}")
        print(f"  reason: {args.reason}")
        print(f"  table : {table}")
        print(f"  audit : {decisions}")
        print("  note  : 旧记录保留（status=superseded），从未删除")
    return 0


def cmd_conflict_resolve(args: argparse.Namespace) -> int:
    """Record a human decision about one conflict. The conflict row is annotated,
    never deleted; ``superseded`` additionally retcons the two assertions."""
    import json
    from pathlib import Path

    from cbb_stage1.validate.adjudication import resolve_conflict

    work = Path(args.work)
    try:
        result = resolve_conflict(
            work,
            conflict_id=args.conflict_id,
            choice=args.choice,
            by=args.by,
            reason=args.reason,
            at=args.at,
        )
    except (ValueError, KeyError) as exc:
        print(f"refused: {exc}")
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"conflict {args.conflict_id} -> {args.choice}")
        print(f"  by    : {args.by}")
        print(f"  reason: {args.reason}")
        print("  note  : 冲突记录保留（只标状态），从未删除")
    return 0


def cmd_quarantine_adjudicate(args: argparse.Namespace) -> int:
    """Record a human decision about one quarantined item. The item is annotated,
    never deleted — the record of what was quarantined is the evidence it happened."""
    import json
    from pathlib import Path

    from cbb_stage1.validate.adjudication import adjudicate_quarantine

    work = Path(args.work)
    try:
        result = adjudicate_quarantine(
            work,
            item_id=args.item_id,
            decision=args.decision,
            by=args.by,
            reason=args.reason,
            at=args.at,
        )
    except (ValueError, KeyError) as exc:
        print(f"refused: {exc}")
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"quarantine {args.item_id} -> {args.decision}")
        print(f"  by    : {args.by}")
        print(f"  reason: {args.reason}")
        print("  note  : 隔离项保留（只标裁决），从未删除")
    return 0


def cmd_model_call(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from cbb_stage1.config import Config
    from cbb_stage1.model_gateway import build_model_gateway

    work = Path(args.work)
    cfg = Config(
        book_id="model-call",
        book_path=Path(args.book or "book.txt"),
        out_dir=Path(args.out or "artifacts/box"),
        work_dir=work,
        provider=args.provider,
        model=args.model or "model",
    )
    gateway = build_model_gateway(cfg, work)
    if gateway is None:
        print("BLOCKED: provider is mock/unconfigured; no model call dispatched")
        return 2
    result = gateway.call(args.prompt, role=args.role, system=args.system or "")
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result.__dict__, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": result.status,
                "role": result.role,
                "text": result.text,
                "error": result.error,
                "provider": result.provider,
                "model": result.model,
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.status == "ok" else 2


def cmd_model_stats(args: argparse.Namespace) -> int:
    import json

    from cbb_stage1.model_ops import summarise_calls

    result = summarise_calls(args.work)
    data = result["data"]
    for key, count in data["by_role_status"].items():
        role, _, status = key.partition(":")
        print(f"{role}\t{status}\t{count}")
    print(f"calls={data['calls']}")
    if getattr(args, "json", False):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_model_drain(args: argparse.Namespace) -> int:
    import json

    from cbb_stage1.model_ops import drain_queue

    result = drain_queue(args.work, queue=getattr(args, "queue", None))
    for row in result["data"]["pending"]:
        print(json.dumps(row, ensure_ascii=False))
    return 0


def cmd_model_reply(args: argparse.Namespace) -> int:
    from pathlib import Path

    from cbb_stage1.agents.host_queue import HostQueueClient

    queue = Path(args.queue) if args.queue else Path(args.work) / "plan" / "host-model"
    client = HostQueueClient(queue)
    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if text is None:
        raise SystemExit("model-reply requires --text or --file")
    client.reply(args.id, text, status=args.status, error=args.error)
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    from cbb_stage1.orchestrator.smoke import smoke

    smoke(Path(args.work))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cbb-stage1")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run extraction pipeline")
    p.add_argument("--config")
    p.add_argument("--book")
    p.add_argument("--book-id", default="book")
    p.add_argument("--out", default="artifacts/box")
    p.add_argument("--work", default="artifacts/work")
    p.add_argument("--chapters", default="1-3")
    p.add_argument("--live", action="store_true")
    p.add_argument("--provider")
    p.add_argument("--model")
    p.add_argument("--nli-provider")
    p.add_argument("--nli-model")
    p.add_argument(
        "--narrator-reliability",
        choices=["reliable", "unreliable", "unknown"],
        help=(
            "D2: is this book's narrator trustworthy? 'unreliable' "
            "downgrades narration-only assertions to claim_layer=statement"
        ),
    )
    p.add_argument("--box", action="store_true", help="build box v2.2 after the run")
    p.add_argument("--series-id")
    p.add_argument("--volume-no", type=int)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("build-box", help="build a box v2.2 from work/")
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--series-id")
    p.add_argument("--volume-no", type=int)
    p.set_defaults(func=cmd_build_box)

    p = sub.add_parser("lint-box", help="validate a box")
    p.add_argument("box")
    p.set_defaults(func=cmd_lint_box)

    p = sub.add_parser("acceptance", help="verify the 07 acceptance matrix")
    p.add_argument("matrix", nargs="?", default="acceptance/07.yaml")
    p.add_argument("--run", action="store_true")
    p.set_defaults(func=cmd_acceptance)

    p = sub.add_parser("model-policy", help="write work/plan/model-policy.json for the skill agent")
    p.add_argument("--work", required=True)
    p.add_argument("--provider")
    p.add_argument("--model")
    p.add_argument("--transport", choices=("host", "mcp", "http", "auto"), default="host")
    p.add_argument("--host-queue", help="1 or a path relative to work/")
    p.add_argument("--mcp-command")
    p.add_argument("--mcp-tool", default="model_generate")
    p.add_argument("--embedding-base-url")
    p.add_argument("--embedding-model")
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--token-budget", type=int)
    p.add_argument("--role-models", help="JSON object role -> model")
    p.add_argument(
        "--session-model-label",
        help="who serves the main reasoning (host transport cannot probe it)",
    )
    p.add_argument("--judge-base-url", help="B4 judge endpoint (must differ from main)")
    p.add_argument("--judge-model", help="B4 judge model id")
    p.add_argument("--nli-base-url", help="D11 dedicated NLI endpoint")
    p.add_argument("--nli-model", help="D11 dedicated NLI model id")
    p.add_argument(
        "--whitebox-backend",
        action="store_true",
        help="backend can return token logprobs (unlocks B3/B4-whitebox)",
    )
    p.set_defaults(func=cmd_model_policy)

    # ── D9 supersede (human decision) ───────────────────────────────
    p = sub.add_parser(
        "supersede",
        help="mark one assertion superseded by another (human decision, never automatic)",
    )
    p.add_argument("--work", required=True)
    p.add_argument("--loser", required=True, help="assertion id being replaced")
    p.add_argument("--winner", required=True, help="assertion id that replaces it")
    p.add_argument("--by", required=True, help="who decided")
    p.add_argument("--reason", required=True, help="why (goes into decisions.jsonl)")
    p.add_argument("--at", help="ISO timestamp (default: now)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_supersede)

    p = sub.add_parser("conflict-resolve", help="record a human decision about a conflict")
    p.add_argument("--work", required=True)
    p.add_argument("--conflict-id", required=True)
    p.add_argument(
        "--choice", required=True,
        choices=["left", "right", "merge", "disputed", "superseded"],
    )
    p.add_argument("--by", required=True, help="who decided (I4 human gate)")
    p.add_argument("--reason", required=True)
    p.add_argument("--at", default="")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_conflict_resolve)

    p = sub.add_parser("quarantine-adjudicate", help="record a human decision about a quarantined item")
    p.add_argument("--work", required=True)
    p.add_argument("--item-id", required=True)
    p.add_argument("--decision", required=True, choices=["keep", "drop", "retain"])
    p.add_argument("--by", required=True, help="who decided")
    p.add_argument("--reason", required=True)
    p.add_argument("--at", default="")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_quarantine_adjudicate)


    # ── D5 lexicon evolution ────────────────────────────────────────
    p = sub.add_parser(
        "lexicon-list",
        help="show unregistered-predicate proposals awaiting a decision",
    )
    p.add_argument("--work", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_lexicon_list)

    p = sub.add_parser(
        "lexicon-approve",
        help="approve one proposal: register it, then report units to re-scan",
    )
    p.add_argument("--work", required=True)
    p.add_argument("--predicate", required=True)
    p.add_argument(
        "--by",
        required=True,
        help="who approves it; I4 forbids predicates grown from data without a human gate",
    )
    p.add_argument("--library", help="override the library recorded on the proposal")
    p.add_argument("--lexicon", help="path to lexicon/predicates.json")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_lexicon_approve)

    p = sub.add_parser("model-call", help="agent-dispatched single model call through the gateway")
    p.add_argument("--work", required=True)
    p.add_argument("--provider", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--role", default="assistant")
    p.add_argument("--prompt", required=True)
    p.add_argument("--system")
    p.add_argument("--out")
    p.add_argument("--book")
    p.set_defaults(func=cmd_model_call)

    p = sub.add_parser("model-stats", help="summarize work/plan/model-calls.jsonl")
    p.add_argument("--work", required=True)
    p.set_defaults(func=cmd_model_stats)

    p = sub.add_parser("model-drain", help="host agent: list pending model requests")
    p.add_argument("--work", default="artifacts/work")
    p.add_argument("--queue")
    p.set_defaults(func=cmd_model_drain)

    p = sub.add_parser("model-reply", help="host agent: answer a pending model request")
    p.add_argument("--work", default="artifacts/work")
    p.add_argument("--queue")
    p.add_argument("--id", required=True)
    p.add_argument("--text")
    p.add_argument("--file")
    p.add_argument("--status", default="ok")
    p.add_argument("--error")
    p.set_defaults(func=cmd_model_reply)

    p = sub.add_parser("doctor", help="pre-flight dependency audit (run before a live run)")
    p.add_argument("--book")
    p.add_argument("--work")
    p.add_argument("--out")
    p.add_argument("--embedding-base-url")
    p.add_argument("--llm-base-url", help="EXT-01 chat endpoint (empty when using the host queue)")
    p.add_argument("--llm-model", help="EXT-01 model id expected on that endpoint")
    p.add_argument("--nli-provider", help="EXT-02 nli provider (disabled|openai|ollama)")
    p.add_argument("--nli-model", help="EXT-02 nli model id")
    p.add_argument("--uq-head-checkpoint", help="EXT-06 B5 UQ head artifact")
    p.add_argument("--uq-train-data", help="EXT-07 labelled JSONL for training the UQ head")
    p.add_argument("--max-tokens", type=int, help="EXT-10 per-call output ceiling")
    p.add_argument("--token-budget", type=int, help="EXT-10 per-run token ceiling")
    p.add_argument("--max-workers", type=int, help="EXT-11/12 concurrency ceiling")
    p.add_argument("--skip", help="comma-separated probes to mark N/A")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("smoke", help="offline mock smoke")
    p.add_argument("--work", default="artifacts/work")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("wiring", help="static entrypoint reachability gate")
    p.add_argument("--check", action="store_true")
    p.set_defaults(func=cmd_wiring)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
