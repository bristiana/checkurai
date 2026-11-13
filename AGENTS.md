# Instructions for coding agents in this repo

Read [CONTRIBUTING.md](CONTRIBUTING.md) before you write, commit or push anything. Its rules are enforced by hooks and CI.

**Hard rules:**
- Commit only work written by this team at the event. Never copy code, text or notes from other repositories.
- Keep research notes, personal notes and anything that isn't project code or project docs out of the repo: put them in `.private/` (gitignored) or outside the clone.
- **Never commit secrets** (tokens, keys, passwords, `.env`), personal data (emails, phone numbers, personal paths, IPs), data, models, logs, videos, 3D-print files or backup files.
- **Never bypass the checks:** no `--no-verify`, no editing `tools/repo_guard.py` or `.githooks/` to get a commit through. If a check blocks you, fix the content or ask the human.
- **Stage explicit paths** (`git add <files>`). Run `git status` and `git diff --cached` before committing.
- **Never rewrite history or force-push** without the human's explicit approval.
- Commit with the clone's configured noreply identity. Don't change `user.email`.
- Several people and agents share the Intel laptop and these clones. Before committing, check `git status` for changes that aren't yours and leave them alone.

The project plan is [docs/PLAN.md](docs/PLAN.md).
