from __future__ import annotations

import argparse
import json
import sys
import time

from .clients.http import ApiError
from .config import AppConfig
from .coordinator import Coordinator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Koala Science multi-agent coordinator")
    parser.add_argument("--env", default=".env", help="Path to env file")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show configuration and agent status")
    sub.add_parser("preflight", help="Check readiness and configuration risks")
    sub.add_parser("export-memory", help="Export memory database summaries to Markdown")
    sub.add_parser("update-profiles", help="Set each agent profile GitHub repo and description")

    scan = sub.add_parser("scan", help="Score live paper opportunities without posting")
    scan.add_argument("--limit", type=int, default=20)
    scan.add_argument("--domain", default=None)

    run_once = sub.add_parser("run-once", help="Execute positive-EV comment/reply actions")
    run_once.add_argument("--limit", type=int, default=20)
    run_once.add_argument("--domain", default=None)
    run_once.add_argument("--max-actions", type=int, default=None)

    deliberate = sub.add_parser("deliberate", help="Submit eligible verdicts")
    deliberate.add_argument("--limit", type=int, default=50)
    deliberate.add_argument("--domain", default=None)
    deliberate.add_argument("--max-actions", type=int, default=10)

    loop = sub.add_parser("loop", help="Continuously run comments/replies and verdicts")
    loop.add_argument("--limit", type=int, default=50)
    loop.add_argument("--domain", default=None)
    loop.add_argument("--max-comment-actions", type=int, default=None)
    loop.add_argument("--max-verdict-actions", type=int, default=10)
    loop.add_argument("--interval", type=int, default=None)
    loop.add_argument("--iterations", type=int, default=0, help="0 means run until interrupted")

    args = parser.parse_args(argv)
    config = AppConfig.from_env(args.env)

    try:
        coordinator = Coordinator(config)
        if args.command == "status":
            print_json(coordinator.status())
            return 0
        if args.command == "preflight":
            result = coordinator.preflight()
            print_json(result)
            return 0 if result["ok"] else 1
        if args.command == "export-memory":
            print_json(coordinator.export_memory())
            return 0
        if args.command == "update-profiles":
            print_json(coordinator.update_profiles())
            return 0
        if args.command == "scan":
            decisions = coordinator.scan(limit=args.limit, domain=args.domain)
            print_json(
                [
                    {
                        "agent": decision.agent.name,
                        "role": decision.agent.public_role,
                        "paper_id": decision.paper.id,
                        "title": decision.paper.title,
                        "status": decision.paper.status,
                        "action": decision.action,
                        "ev": round(decision.ev, 3),
                        "reason": decision.reason,
                        "roles": decision.roles,
                        "features": decision.features,
                    }
                    for decision in decisions[:30]
                ]
            )
            return 0
        if args.command == "run-once":
            print_json(
                coordinator.run_once(
                    limit=args.limit, domain=args.domain, max_actions=args.max_actions
                )
            )
            return 0
        if args.command == "deliberate":
            print_json(
                coordinator.deliberate(
                    limit=args.limit, domain=args.domain, max_actions=args.max_actions
                )
            )
            return 0
        if args.command == "loop":
            interval = args.interval or config.loop_interval_seconds
            iterations = 0
            while True:
                coordinator = Coordinator(AppConfig.from_env(args.env))
                cycle = {
                    "comments": coordinator.run_once(
                        limit=args.limit,
                        domain=args.domain,
                        max_actions=args.max_comment_actions,
                    ),
                    "verdicts": coordinator.deliberate(
                        limit=args.limit,
                        domain=args.domain,
                        max_actions=args.max_verdict_actions,
                    ),
                }
                print_json(cycle)
                sys.stdout.flush()
                iterations += 1
                if args.iterations and iterations >= args.iterations:
                    return 0
                time.sleep(interval)
    except ApiError as exc:
        print(f"Koala/API request failed: {exc}", file=sys.stderr)
        if exc.body:
            print(exc.body[:1000], file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Runtime error: {exc}", file=sys.stderr)
        return 3

    return 1


def print_json(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=to_json))


def to_json(value):
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)
