"""Run a render on the build box, and bring the results home. One command.

Rendering somewhere else is only worth it if it is EASIER than rendering here. Two minutes
of ssh, git and scp typed by hand is not easier than one local command, so the local command
wins every time, at fourteen threads, for the rest of the afternoon. This script makes the
remote path the short one: commit what is uncommitted, push, pull and build there, run, and
copy the outputs back.

    uv run python tools/render_on_box.py examples/cases/26_forecourt.py
    uv run python tools/render_on_box.py -m forecourt.sheet --cwd examples/cases

Configuration comes from the environment, never from this file — a build host is somebody's
private machine and does not belong in a public repository:

    MIRAGE_BOX      user@host for ssh (required)
    MIRAGE_BOX_DIR  the repo's path on that host (required)
    MIRAGE_BOX_ENV  extra exports for the remote shell, e.g. "PATH=$HOME/.local/bin:$PATH"
    MIRAGE_BOX_PULL comma-separated paths to copy back (default: the case outputs directory)

On Windows run this from PowerShell, not Git Bash: MSYS rewrites anything shaped like a
Unix path, so a remote target arrives at scp as C:/Program Files/Git/home/...

Assets that are gitignored but must match — generated decal artwork, whose CJK text falls
back to a bitmap font on a host without the right fonts installed — are pushed by scp, since
git will not carry them.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Only the OUTPUTS by default. Pulling all of docs/gallery drags back the scripts that
# live there too, and a Linux checkout hands them over with LF endings, so every render
# turns into a diff full of files nothing touched.
DEFAULT_PULL = ["examples/cases/outputs/26_forecourt"]
PUSH_ASSETS = ["assets/decals"]


# Git Bash rewrites anything that looks like a Unix path in an argument into a Windows one,
# so a perfectly good remote target becomes "C:/Program Files/Git/home/...". Turning that off
# for our own subprocesses is the documented escape hatch, and it is a no-op everywhere else.
_ENV = {**os.environ, "MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*"}


def sh(*args, **kw):
    r = subprocess.run(args, cwd=str(ROOT), text=True, env=_ENV, **kw)
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}): {' '.join(args)}")
    return r


def out(*args):
    return subprocess.run(args, cwd=str(ROOT), text=True, capture_output=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="what to run remotely, e.g. examples/cases/26_forecourt.py")
    ap.add_argument("--cwd", default=".", help="directory on the box to run from (repo-relative)")
    ap.add_argument("--no-build", action="store_true", help="skip the remote cmake build")
    ap.add_argument("--threads", default="140")
    args = ap.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]     # argparse keeps the separator in a REMAINDER
    if not args.command:
        ap.error("nothing to run")

    box, box_dir = os.environ.get("MIRAGE_BOX"), os.environ.get("MIRAGE_BOX_DIR")
    if not box or not box_dir:
        raise SystemExit("set MIRAGE_BOX (user@host) and MIRAGE_BOX_DIR (repo path there)")
    env = os.environ.get("MIRAGE_BOX_ENV", "PATH=$HOME/.local/bin:$PATH")
    pull = [p for p in os.environ.get("MIRAGE_BOX_PULL", ",".join(DEFAULT_PULL)).split(",") if p]

    if out("git", "status", "--porcelain"):
        raise SystemExit("working tree is dirty — commit first, so the box renders what you\n"
                         "  are about to claim it rendered (git status)")
    print(f"[box] pushing {out('git', 'rev-parse', '--short', 'HEAD')}")
    sh("git", "push", "-q", "origin", "HEAD")

    for a in PUSH_ASSETS:                     # gitignored, generated, must match exactly
        if (ROOT / a).is_dir():
            sh("ssh", "-o", "BatchMode=yes", box, f"mkdir -p {box_dir}/{a}")
            sh("scp", "-q", "-r", str(ROOT / a) + "/.", f"{box}:{box_dir}/{a}/")

    # A render writes into docs/gallery, so the box's copy is always locally modified;
    # discard it rather than letting the pull abort on a file we are about to overwrite.
    # Export FIRST: a box that reaches github through a proxy needs it set before the pull,
    # and a non-interactive ssh does not read the shell profile that normally provides it.
    # Also retry the pull once — a transient "flush packet" from the remote should not cost a
    # whole render, and worse, should not leave a STALE image on disk for the next step to
    # score and report as if it were new.
    steps = [f"cd {box_dir}", f"export {env} MIRAGE_THREADS={args.threads}",
             "git checkout -- docs/gallery 2>/dev/null || true",
             "git pull --ff-only -q || (sleep 3 && git pull --ff-only -q)"]
    if not args.no_build:
        steps.append("cmake --build core/build -j >/dev/null")
    run = " ".join(args.command)
    steps.append(f"cd {box_dir}/{args.cwd} && uv run python {run}")
    print(f"[box] {run}")
    sh("ssh", "-o", "BatchMode=yes", box, " && ".join(steps))

    for p in pull:
        (ROOT / p).mkdir(parents=True, exist_ok=True)
        sh("scp", "-q", "-r", f"{box}:{box_dir}/{p}/.", str(ROOT / p) + "/")
        print(f"[box] pulled {p}")


if __name__ == "__main__":
    sys.exit(main())
