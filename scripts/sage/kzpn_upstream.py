"""The upstream KZPN sources that the fixture scripts load, and their provenance.

The fixtures are generated from a clean checkout of

    https://github.com/antieau/kzpn  at commit ed381b4bf8df9522a12829ee0cbc9a2374949231

(set ``KZPN_DIR`` to it). ``kzpn_provenance`` records which files were actually
read: their SHA-256, whether they are byte-identical to the files of that
commit, and, if ``KZPN_DIR`` is a git checkout, its HEAD and whether the three
files are unmodified there. Plain Python (no Sage), imported by both scripts.
"""

import os
import subprocess

UPSTREAM_REPOSITORY = "https://github.com/antieau/kzpn"
UPSTREAM_COMMIT = "ed381b4bf8df9522a12829ee0cbc9a2374949231"
KZPN_FILES = ("prismatic_envelope.sage", "precision.py", "matrix_functions.py")
# SHA-256 of the three files at UPSTREAM_COMMIT.
UPSTREAM_SHA256 = {
    "prismatic_envelope.sage": "1275d306c46e97ee3bc951c7d482bb6d009df3d92baed1207f150b68e288ed04",
    "precision.py": "ee80940fd83ba646d7a185af1a6f67b910431f7722a39b68e35d960eef7767b4",
    "matrix_functions.py": "bdc2a12ecfa85313fbb6720eaf33177f31a4be0f91dab20b9a285b7e414e63e1",
}


def _git(kzpn_dir, *args):
    try:
        out = subprocess.run(
            ["git", "-C", kzpn_dir] + list(args), capture_output=True, text=True, timeout=60
        )
    except Exception:  # git missing
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state(kzpn_dir):
    """HEAD of the git checkout ``kzpn_dir`` and whether the KZPN files are unmodified there."""
    head = _git(kzpn_dir, "rev-parse", "HEAD")
    if head is None:
        return None
    status = _git(kzpn_dir, "status", "--porcelain", "--", *KZPN_FILES)
    return {"head": head, "kzpn_files_unmodified": status == ""}


def kzpn_provenance(kzpn_dir, sha):
    """Provenance entries for the KZPN files read from ``kzpn_dir`` (``sha``: name -> SHA-256)."""
    matches = all(sha.get(name) == UPSTREAM_SHA256[name] for name in KZPN_FILES)
    return {
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "sha256": dict(UPSTREAM_SHA256),
        },
        "kzpn_sha256": dict(sha),
        "kzpn_matches_upstream_commit": matches,
        "kzpn_git": git_state(kzpn_dir),
    }


def warn_if_not_upstream(prov, log):
    """Print a warning when the files read are not those of the upstream commit."""
    if not prov["kzpn_matches_upstream_commit"]:
        log(
            "WARNING: the KZPN files differ from %s at %s; the fixtures record this "
            "(provenance.kzpn_matches_upstream_commit = false)" % (UPSTREAM_REPOSITORY, UPSTREAM_COMMIT[:12])
        )
