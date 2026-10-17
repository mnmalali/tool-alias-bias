# Manual GitHub release

The Git repository contains source, frozen inputs, compact outputs, and
verification tools. The full raw request/response/attempt record is a separate
GitHub release asset and is intentionally ignored by Git.

Before publishing, run `python scripts/verify_release.py` and then inspect the
release asset listed in `results/observed-study/raw-data.md`. Do not add the
ignored `release-assets/` directory to a normal Git commit.

After the repository is pushed, create a GitHub release with the tag `v0.1.0`
and upload the prepared `.tar.gz` file from `release-assets/`. Copy the release
asset URL into `results/observed-study/raw-data.md`, commit that final link, and
publish or update the GitHub release. The asset checksum recorded in that file
must match the uploaded file.

This release package records a partial run and unreviewed labels. Do not remove
the failure accounting, limitation language, or raw-data provenance when
publishing it.
