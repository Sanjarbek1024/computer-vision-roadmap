#!/usr/bin/env python
"""Read the analytics database and answer questions about past sessions.

    python tools/report.py --list                  # every recorded session
    python tools/report.py                         # detail on the newest one
    python tools/report.py --session session_x     # detail on a specific one
    python tools/report.py --charts                # + PNG charts next to it
    python tools/report.py --compare run_a run_b   # two runs side by side

This is the payoff for writing to SQLite instead of just dumping a video: the
questions below are plain SQL over every session ever recorded.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_DIR / "outputs" / "analytics.db"


def connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise SystemExit(f"No database at {db_path}. Run the pipeline first.")
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    return db


def format_seconds(value: float) -> str:
    value = float(value or 0)
    minutes, seconds = divmod(int(value), 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


# ------------------------------------------------------------------- listing
def list_sessions(db: sqlite3.Connection) -> None:
    rows = db.execute(
        "SELECT session_id, started_at, source, processed_frames, "
        "       students_detected, peak_present, model, tracker "
        "FROM sessions ORDER BY started_at DESC"
    ).fetchall()

    if not rows:
        raise SystemExit("The database has no sessions yet.")

    print(f"{'session':<26}{'started':<22}{'frames':>8}{'students':>10}{'peak':>6}  model")
    print("-" * 88)
    for row in rows:
        print(f"{row['session_id']:<26}{(row['started_at'] or ''):<22}"
              f"{row['processed_frames']:>8}{row['students_detected']:>10}"
              f"{row['peak_present']:>6}  {row['model']}")


def latest_session(db: sqlite3.Connection) -> str:
    row = db.execute(
        "SELECT session_id FROM sessions ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        raise SystemExit("The database has no sessions yet.")
    return row["session_id"]


# -------------------------------------------------------------------- detail
def session_report(db: sqlite3.Connection, session_id: str) -> None:
    session = db.execute(
        "SELECT * FROM sessions WHERE session_id=?", (session_id,)
    ).fetchone()
    if session is None:
        raise SystemExit(f"No session called {session_id}.")

    print("=" * 78)
    print(f"SESSION {session_id}")
    print("=" * 78)
    print(f"  source     : {session['source']}")
    print(f"  recorded   : {session['started_at']} -> {session['ended_at']}")
    print(f"  video      : {session['width']}x{session['height']} @ "
          f"{session['fps']:.2f} fps, {session['processed_frames']} frames analysed")
    print(f"  detector   : {session['model']} imgsz={session['imgsz']} "
          f"conf={session['conf']}")
    print(f"  tracker    : {session['tracker']}")
    print(f"  students   : {session['students_detected']} confirmed, "
          f"{session['tracks_discarded']} noise tracks discarded")
    print(f"  peak       : {session['peak_present']} people in the room at once")

    # ---- attendance -------------------------------------------------------
    students = db.execute(
        "SELECT * FROM students WHERE session_id=? ORDER BY student_id", (session_id,)
    ).fetchall()

    if students:
        print("\nATTENDANCE")
        print(f"  {'id':>4}  {'in':>8}  {'out':>8}  {'duration':>9}  {'seen':>6}"
              f"  {'pred':>6}  {'qual':>5}  {'conf':>5}  {'re-id':>6}  home zone")
        print("  " + "-" * 92)
        for row in students:
            print(f"  S{row['student_id']:>3}  "
                  f"{format_seconds(row['first_seen_s']):>8}  "
                  f"{format_seconds(row['last_seen_s']):>8}  "
                  f"{format_seconds(row['duration_s']):>9}  "
                  f"{row['visible_frames']:>6}  {row['predicted_frames']:>6}  "
                  f"{row['tracking_quality']:>5.2f}  {row['avg_confidence']:>5.2f}  "
                  f"{row['reidentifications']:>6}  {row['home_zone']}")

        total = sum(row["duration_s"] for row in students)
        print(f"\n  total student-time: {format_seconds(total)}  |  "
              f"average stay: {format_seconds(total / len(students))}")

    # ---- movement ---------------------------------------------------------
    movers = db.execute(
        "SELECT student_id, moving_s, distance_px, duration_s FROM students "
        "WHERE session_id=? AND duration_s > 1 "
        "ORDER BY moving_s DESC LIMIT 5", (session_id,)
    ).fetchall()

    if movers:
        print("\nMOST ACTIVE")
        for row in movers:
            share = 100.0 * row["moving_s"] / max(0.1, row["duration_s"])
            print(f"  S{row['student_id']:>3}  moving {format_seconds(row['moving_s']):>7}"
                  f"  ({share:4.1f}% of their time)  "
                  f"travelled {row['distance_px']:.0f}px")

    # ---- zones ------------------------------------------------------------
    zones = db.execute(
        "SELECT zone, SUM(duration_s) AS total, COUNT(*) AS visits "
        "FROM zone_visits WHERE session_id=? GROUP BY zone "
        "ORDER BY total DESC", (session_id,)
    ).fetchall()

    if zones:
        print("\nZONE OCCUPANCY")
        peak = max(row["total"] for row in zones) or 1
        for row in zones:
            bar = "#" * int(28 * row["total"] / peak)
            print(f"  {row['zone']:<16}{format_seconds(row['total']):>8}  "
                  f"{row['visits']:>3} visits  {bar}")

    # ---- occupancy over time ---------------------------------------------
    buckets = db.execute(
        "SELECT bucket_start_s, avg_present, peak_present FROM occupancy "
        "WHERE session_id=? ORDER BY bucket_start_s", (session_id,)
    ).fetchall()

    if buckets:
        print("\nOCCUPANCY OVER TIME")
        peak = max(row["peak_present"] for row in buckets) or 1
        for row in buckets:
            bar = "#" * int(30 * row["avg_present"] / peak)
            print(f"  {format_seconds(row['bucket_start_s']):>7}  "
                  f"avg {row['avg_present']:>5.1f}  peak {row['peak_present']:>2}  {bar}")

    # ---- events -----------------------------------------------------------
    events = db.execute(
        "SELECT type, COUNT(*) AS n FROM events WHERE session_id=? "
        "GROUP BY type ORDER BY n DESC", (session_id,)
    ).fetchall()

    if events:
        print("\nEVENTS")
        for row in events:
            print(f"  {row['type']:<18}{row['n']:>5}")

    print("=" * 78)


# ------------------------------------------------------------------ compare
def compare(db: sqlite3.Connection, session_ids: list[str]) -> None:
    print(f"{'metric':<26}" + "".join(f"{sid[:18]:>20}" for sid in session_ids))
    print("-" * (26 + 20 * len(session_ids)))

    rows = []
    for sid in session_ids:
        session = db.execute("SELECT * FROM sessions WHERE session_id=?",
                             (sid,)).fetchone()
        if session is None:
            raise SystemExit(f"No session called {sid}.")

        stats = db.execute(
            "SELECT COUNT(*) AS n, AVG(duration_s) AS avg_dur, "
            "       AVG(tracking_quality) AS avg_q, SUM(reidentifications) AS reids, "
            "       AVG(avg_confidence) AS avg_conf "
            "FROM students WHERE session_id=?", (sid,)
        ).fetchone()

        rows.append({
            "students": session["students_detected"],
            "peak_present": session["peak_present"],
            "discarded": session["tracks_discarded"],
            "avg_stay_s": round(stats["avg_dur"] or 0, 1),
            "avg_quality": round(stats["avg_q"] or 0, 3),
            "re_ids": stats["reids"] or 0,
            "avg_confidence": round(stats["avg_conf"] or 0, 3),
            "frames": session["processed_frames"],
            "tracker": (session["tracker"] or "").split("/")[-1],
        })

    for metric in ("frames", "students", "peak_present", "discarded", "avg_stay_s",
                   "avg_quality", "re_ids", "avg_confidence", "tracker"):
        print(f"{metric:<26}" + "".join(f"{str(row[metric]):>20}" for row in rows))


# -------------------------------------------------------------------- charts
def charts(db: sqlite3.Connection, session_id: str, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping charts")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = db.execute(
        "SELECT bucket_start_s, avg_present, peak_present, avg_moving FROM occupancy "
        "WHERE session_id=? ORDER BY bucket_start_s", (session_id,)
    ).fetchall()
    students = db.execute(
        "SELECT student_id, first_seen_s, last_seen_s, duration_s, moving_s "
        "FROM students WHERE session_id=? ORDER BY student_id", (session_id,)
    ).fetchall()

    if not buckets and not students:
        print("nothing to chart for this session")
        return

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    fig.suptitle(f"Classroom analytics - {session_id}", fontsize=13)

    if buckets:
        times = [row["bucket_start_s"] / 60.0 for row in buckets]
        axes[0].plot(times, [row["avg_present"] for row in buckets],
                     label="average present", linewidth=2)
        axes[0].plot(times, [row["peak_present"] for row in buckets],
                     label="peak", linestyle="--", linewidth=1)
        axes[0].fill_between(times, [row["avg_moving"] for row in buckets],
                             alpha=0.3, label="moving")
        axes[0].set_xlabel("minutes")
        axes[0].set_ylabel("people")
        axes[0].set_title("Occupancy over time")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    if students:
        # Presence timeline: one bar per student, from entry to exit.
        for row in students:
            axes[1].barh(row["student_id"],
                         (row["last_seen_s"] - row["first_seen_s"]) / 60.0,
                         left=row["first_seen_s"] / 60.0, height=0.7)
        axes[1].set_xlabel("minutes")
        axes[1].set_ylabel("student")
        axes[1].set_title("Presence timeline")
        axes[1].grid(alpha=0.3, axis="x")

        ids = [f"S{row['student_id']:02d}" for row in students]
        axes[2].bar(ids, [row["duration_s"] / 60.0 for row in students],
                    label="time in room")
        axes[2].bar(ids, [row["moving_s"] / 60.0 for row in students],
                    label="moving")
        axes[2].set_ylabel("minutes")
        axes[2].set_title("Time in room vs time moving")
        axes[2].legend()
        axes[2].grid(alpha=0.3, axis="y")
        plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out_dir / "charts.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"charts saved -> {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(DEFAULT_DB), help="analytics database")
    parser.add_argument("--session", help="session id (default: the newest)")
    parser.add_argument("--list", action="store_true", help="list sessions and exit")
    parser.add_argument("--compare", nargs="+", metavar="SESSION",
                        help="compare two or more sessions")
    parser.add_argument("--charts", action="store_true", help="also save PNG charts")
    args = parser.parse_args(argv)

    db = connect(Path(args.db))
    try:
        if args.list:
            list_sessions(db)
            return 0

        if args.compare:
            compare(db, args.compare)
            return 0

        session_id = args.session or latest_session(db)
        session_report(db, session_id)

        if args.charts:
            charts(db, session_id, Path(args.db).parent / session_id)
    finally:
        db.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
