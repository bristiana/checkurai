#!/usr/bin/env python3
"""Repo guard: keeps every commit clean (no secrets, no personal data, no big or forbidden files).

Run by the git hooks in .githooks/ and by CI. Standard library only.

    python tools/repo_guard.py staged          # pre-commit: staged changes
    python tools/repo_guard.py msg FILE        # commit-msg: the commit message
    python tools/repo_guard.py push REMOTE URL # pre-push: every commit about to be pushed (refs on stdin)
    python tools/repo_guard.py history         # every commit on every branch (run before going public)
    python tools/repo_guard.py tree            # every tracked file as it is now (CI)

Extra local patterns (one regex per line, '#' comments) are read from .private/denylist.txt
when it exists. That folder is gitignored, so the list itself never reaches the repo.
Exit code 1 = blocked. There is no override flag on purpose; CI re-checks everything on push.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 2 * 1024 * 1024  # 2 MB per file
ALLOWED_EMAIL = re.compile(r"@users\.noreply\.github\.com$", re.I)
SELF = "tools/repo_guard.py"

FORBIDDEN_PATHS = [
    (r"(^|/)\.private/", "private notes folder"),
    (r"(^|/)\.env(\.[\w-]+)?$", "environment file (secrets)"),
    (r"(^|/)(id_rsa|id_ed25519|id_ecdsa|id_dsa)(\.pub)?$", "SSH key"),
    (r"_deploy(\.pub)?$|\.(pem|key|p12|pfx|keystore|jks)$", "key / certificate file"),
    (r"(^|/)\.?modal\.toml$|(^|/)\.netrc$|(^|/)\.git-credentials$|credentials\.json$", "credentials file"),
    (r"(^|/)(data|datasets?|exports?|outputs|models|checkpoints|snapshots|logs)/", "data / model / log folder"),
    (r"\.(pt|pth|ckpt|safetensors|onnx|bin|tflite|engine|npz|npy|pkl|parquet|arrow|h5)$", "model / dataset file"),
    (r"\.(mp4|mkv|avi|mov|webm|zip|tar|gz|tgz|7z|rar)$", "video / archive"),
    (r"\.(3mf|stl|step|stp|gcode)$", "3D print file (licence)"),
    (r"(^|/)(belt/)?build/|(^|/)\.pio/", "build output"),
    (r"\.bak(-[\w.-]+)?$|~$|\.orig$|\.swp$", "backup / editor file"),
]

SECRETS = [
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
    (r"\bhf_[A-Za-z0-9]{30,}", "Hugging Face token"),
    (r"\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{20,}", "GitHub token"),
    (r"\b(ak|as)-[A-Za-z0-9]{16,}", "Modal token"),
    (r"\bAKIA[0-9A-Z]{16}\b|aws_secret_access_key\s*[:=]", "AWS key"),
    (r"\bsk-(ant-|proj-)?[A-Za-z0-9_-]{20,}", "API key (sk-...)"),
    (r"\bAIza[0-9A-Za-z_-]{35}\b", "Google API key"),
    (r"\bxox[abprs]-[A-Za-z0-9-]{10,}", "Slack token"),
    (r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}", "JWT"),
    (r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)\s*[:=]\s*['\"][^'\"\s]{6,}['\"]", "hard-coded credential"),
    (r"(?i)(sudo\s+-S|echo\s+\S+\s*\|\s*sudo)", "password piped into sudo"),
    (r"(?i)(https?://)[^/\s:@]+:[^/\s:@]+@", "credentials in URL"),
]

PRIVACY = [
    (r"(?<![\w.+-])[\w.+-]+@(?!users\.noreply\.github\.com)(?!example\.(com|org))[A-Za-z0-9-]+\.[A-Za-z]{2,}(?![\w-])", "email address"),
    (r"(?<![\d.])(10\.\d{1,3}|192\.168|172\.(1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?![\d.])", "private IP address"),
    (r"(?<!\w)\+\d{2,3}[ -]?\d{2,4}[ -]?\d{3}[ -]?\d{3,4}(?!\d)", "phone number"),
    (r"(?i)[a-z]:\\+users\\+(?![<%$])[^\\\s%]+", "personal Windows path"),
    (r"/home/(?!intel\b)[a-z][\w-]*|/Users/(?!Shared\b)[A-Za-z][\w-]*", "personal home path"),
]
# git remote host aliases look like emails ("git@github.com:..."); allow them
EMAIL_ALLOW = re.compile(r"git@[\w.-]+[:/]", re.I)


def git(*args: str, input: str | None = None) -> str:
    res = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", input=input)
    if res.returncode != 0:
        raise SystemExit(f"repo_guard: git {' '.join(args)} failed: {res.stderr.strip()}")
    return res.stdout


def root() -> Path:
    return Path(git("rev-parse", "--show-toplevel").strip())


def load_denylist() -> list[tuple[re.Pattern, str]]:
    f = root() / ".private" / "denylist.txt"
    if not f.exists():
        return []
    pats = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            pats.append((re.compile(line, re.I), "local denylist term"))
    return pats


class Report:
    def __init__(self) -> None:
        self.problems: list[str] = []

    def add(self, where: str, what: str, sample: str = "") -> None:
        s = sample.strip()
        if len(s) > 60:
            s = s[:57] + "..."
        # Never echo a full secret back to the terminal.
        if s and what not in ("email address", "personal Windows path", "personal home path", "local denylist term"):
            s = s[:6] + "***" if len(s) > 6 else "***"
        if what == "local denylist term":
            s = "(hidden)"
        self.problems.append(f"  {where}: {what}" + (f"  [{s}]" if s else ""))

    def done(self, context: str) -> int:
        if not self.problems:
            return 0
        print(f"\nrepo_guard BLOCKED ({context}). Fix these, then retry:", file=sys.stderr)
        for p in dict.fromkeys(self.problems):
            print(p, file=sys.stderr)
        print("\nRules: CONTRIBUTING.md. Secrets go in env vars / tool profiles; personal notes in .private/.",
              file=sys.stderr)
        return 1


def check_path(rep: Report, path: str, where: str) -> None:
    for pat, why in FORBIDDEN_PATHS:
        if re.search(pat, path, re.I):
            rep.add(where, f"forbidden file ({why})", "")
            return


def check_text(rep: Report, text: str, where: str, deny, is_self: bool = False) -> None:
    rules = ([] if is_self else SECRETS + PRIVACY) + deny
    for pat, why in rules:
        rx = pat if isinstance(pat, re.Pattern) else re.compile(pat)
        for m in rx.finditer(text):
            if why == "email address" and EMAIL_ALLOW.search(text[max(0, m.start() - 4):m.end() + 1]):
                continue
            rep.add(where, why, m.group(0))


def added_lines(diff: str):
    """Yield (path, lineno, text) for every added line in a -U0 unified diff."""
    path, ln = None, 0
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            ln = int(m.group(1)) if m else 0
        elif line.startswith("+") and path:
            yield path, ln, line[1:]
            ln += 1


def check_diff(rep: Report, diff: str, deny, prefix: str = "") -> None:
    for path, ln, text in added_lines(diff):
        check_text(rep, text, f"{prefix}{path}:{ln}", deny, is_self=(path == SELF))


def check_commit(rep: Report, sha: str, deny) -> None:
    short = sha[:7]
    ae, ce = git("log", "-1", "--format=%ae%n%ce", sha).split("\n")[:2]
    for e in {ae, ce}:
        if not ALLOWED_EMAIL.search(e):
            rep.add(f"commit {short}", "author/committer email is not a GitHub noreply address", e)
    check_text(rep, git("log", "-1", "--format=%B", sha), f"commit {short} message", deny)
    for line in git("diff-tree", "-r", "--root", "--no-commit-id", "-M", "--raw", sha).splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) < 5 or parts[4].startswith("D"):
            continue
        path = path.split("\t")[-1]
        check_path(rep, path, f"commit {short} {path}")
        blob = parts[3]
        if set(blob) != {"0"} and int(git("cat-file", "-s", blob).strip()) > MAX_BYTES:
            rep.add(f"commit {short} {path}", f"file larger than {MAX_BYTES // 1024 // 1024} MB")
    check_diff(rep, git("show", "-U0", "--no-color", "--format=", "--root", "-M", sha), deny, f"commit {short} ")


def mode_staged() -> int:
    rep, deny = Report(), load_denylist()
    if not deny:
        print("repo_guard: note: .private/denylist.txt not found (local term list skipped)", file=sys.stderr)
    for line in git("diff", "--cached", "--name-status", "-M", "--diff-filter=ACMR").splitlines():
        path = line.split("\t")[-1]
        check_path(rep, path, path)
        size = git("cat-file", "-s", f":{path}").strip()
        if size.isdigit() and int(size) > MAX_BYTES:
            rep.add(path, f"file larger than {MAX_BYTES // 1024 // 1024} MB")
    check_diff(rep, git("diff", "--cached", "-U0", "--no-color", "-M", "--diff-filter=ACMR"), deny)
    return rep.done("pre-commit")


def mode_msg(msgfile: str) -> int:
    rep = Report()
    text = "\n".join(l for l in Path(msgfile).read_text(encoding="utf-8").splitlines() if not l.startswith("#"))
    check_text(rep, text, "commit message", load_denylist())
    return rep.done("commit-msg")


def mode_push(remote: str) -> int:
    rep, deny, zero = Report(), load_denylist(), "0" * 40
    for line in sys.stdin.read().splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[1] == zero:
            continue
        local_sha, remote_sha = parts[1], parts[3]
        rng = [f"{remote_sha}..{local_sha}"] if remote_sha != zero else [local_sha, "--not", f"--remotes={remote}"]
        for sha in git("rev-list", *rng).split():
            check_commit(rep, sha, deny)
    return rep.done("pre-push")


def mode_history() -> int:
    rep, deny = Report(), load_denylist()
    shas = git("rev-list", "--all").split()
    for sha in shas:
        check_commit(rep, sha, deny)
    print(f"repo_guard: scanned {len(shas)} commits on all branches", file=sys.stderr)
    return rep.done("history")


def mode_tree() -> int:
    rep, deny, base = Report(), load_denylist(), root()
    files = git("ls-files", "-z").split("\0")
    for path in filter(None, files):
        check_path(rep, path, path)
        p = base / path
        if p.is_file():
            if p.stat().st_size > MAX_BYTES:
                rep.add(path, f"file larger than {MAX_BYTES // 1024 // 1024} MB")
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                check_text(rep, line, f"{path}:{i}", deny, is_self=(path == SELF))
    print(f"repo_guard: scanned {len(files) - 1} tracked files", file=sys.stderr)
    return rep.done("tree")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    mode = argv[0]
    if mode == "staged":
        return mode_staged()
    if mode == "msg" and len(argv) == 2:
        return mode_msg(argv[1])
    if mode == "push" and len(argv) >= 2:
        return mode_push(argv[1])
    if mode == "history":
        return mode_history()
    if mode == "tree":
        return mode_tree()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
