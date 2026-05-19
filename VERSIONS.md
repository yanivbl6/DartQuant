# Sim-version history (`--sim_version N`)

Each row pins a version to the commit it was first run on. The **last row is the current version** — quote it when launching new runs unless the user asks for something else.

See `CLAUDE.md` → "Sim versions" for the bump rules (bug-fix → suffix, major → user-requested only) and what the tag actually affects.

| Version | Commit  | Note |
|--------:|---------|------|
| 74      | cf1248d | current — patch_M8S4 bug part 1 (HEAD of `scalewise` as of 2026-05-19) |

## How to update

When you bump the version (whether a bug-fix suffix like 74 → 741, or a user-requested major like 74 → 75):

1. Set `--sim_version <new>` in the runfile(s) you're about to launch.
2. Append a new row here with the new version, the current `git rev-parse --short HEAD`, and a one-line note (what changed, or what's about to be tested).
3. Commit `VERSIONS.md` together with the code change that motivated the bump, so the commit hash in the row actually contains the new behavior.
