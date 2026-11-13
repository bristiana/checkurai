# Repo rules (CheckurAI, Junction 2026)

This repo becomes **public at submission**. Everything in it, including **every past commit, commit message and author email**, will be public. The rules below are enforced by git hooks and by CI. A blocked commit means: fix it, don't work around it.

## One-time setup (every clone, every person)
```bash
git config core.hooksPath .githooks                                   # turns the checks on
git config user.email "<id>+<github-login>@users.noreply.github.com"  # never a personal email
```
Your noreply address is under GitHub → Settings → Emails ("Keep my email address private").

## The rules
1. **Only our own work.**
   - Everything in this repo is written by this team during Junction 2026 (13–15 Nov).
   - Third-party code comes in through package managers (`pip`, `uv`, `npm`) and is listed in [THIRD_PARTY.md](THIRD_PARTY.md).
   - Never paste code from other repositories.
   - Personal notes, research material and anything not written by us stay **out of the repo**: put them in `.private/` (gitignored) or outside the clone.
2. **No secrets, ever.** That covers tokens (Hugging Face, Modal, GitHub, cloud), SSH/deploy keys, passwords (laptop, Wi-Fi, partner portals) and `.env` files.
   - Credentials live in environment variables or in each tool's own login profile (`modal token new --profile <name>`, `huggingface-cli login`).
   - If a secret was ever committed, **rotate it**. Deleting it in a later commit is not enough.
3. **No personal data.** No emails, phone numbers, personal paths (`C:\Users\<name>`, `/home/<name>`) or IP addresses. Commit only with your GitHub noreply email.
4. **No big or generated files.** No datasets, recordings, model weights or exports (`.pt`, `.ckpt`, `.onnx`, `.bin`, `.safetensors`), videos, archives, build output, logs or snapshots. Back them up outside git. Every file must be under 2 MB.
5. **No licence-restricted assets.** No 3D-print files (`.3mf`, `.stl`, `.step`, `.gcode`): the models we print are personal-use only.
6. **No backup or editor files** (`*.bak`, `*.orig`, `*~`). Git is the backup.
7. **Commit hygiene.**
   - `git add <files>`, not `git add -A`.
   - One topic per commit. The message says what changed and why, in plain words.
   - Don't rewrite or force-push shared branches without agreeing it in the team chat.

## What enforces it
| Where | What | Command |
|---|---|---|
| `pre-commit` | staged files and lines: secrets, personal data, forbidden or big files | `python tools/repo_guard.py staged` |
| `commit-msg` | the commit message | `python tools/repo_guard.py msg <file>` |
| `pre-push` | every commit being pushed, including author/committer emails | `python tools/repo_guard.py push origin <url>` |
| CI (`.github/workflows/guard.yml`) | the whole tree plus every commit on every branch, then Gitleaks | runs on every push and PR |

- **Extra local terms:** each clone can add case-insensitive regexes to `.private/denylist.txt` (gitignored, never shared through the repo). The hooks block those terms too.
- **Before making the repo public:**

  ```bash
  python tools/repo_guard.py tree && python tools/repo_guard.py history
  ```

  Both must pass.
