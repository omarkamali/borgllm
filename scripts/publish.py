#!/usr/bin/env python3
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, List


CHANGELOG_PATH = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"



def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=False)



def get_version_from_pyproject(pyproject_path: Path = PYPROJECT_PATH) -> Optional[str]:
    try:
        for line in pyproject_path.read_text(encoding="utf-8").splitlines():
            m = re.search(r'^version\s*=\s*"(\d+\.\d+\.\d+)"', line)
            if m:
                return m.group(1)
    except FileNotFoundError:
        return None
    return None



def git_is_clean() -> Tuple[bool, str]:
    try:
        # Working tree
        dirty_wc = subprocess.run(["git", "diff", "--quiet", "--exit-code"]).returncode != 0
        # Staged
        dirty_staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--exit-code"]).returncode != 0
        if dirty_wc or dirty_staged:
            status = subprocess.run(["git", "status"], capture_output=True, text=True).stdout
            return False, status
        return True, ""
    except Exception as e:
        return False, str(e)



def git_tag_exists(tag: str) -> bool:
    return subprocess.run(["git", "rev-parse", tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0



SECTION_HEADER_RE = re.compile(r"^## \[(?P<version>\d+\.\d+\.\d+)\] - (?P<date>\d{4}-\d{2}-\d{2})\s*$")



def extract_latest_changelog(changelog_text: str) -> Tuple[str, str]:
    """
    Extract the latest section from a Keep a Changelog-like file.


    Returns (version, section_text) where section_text includes the header and body for the latest version.
    Raises ValueError if no versioned section is found.
    """
    lines = changelog_text.splitlines()
    sections: List[Tuple[str, int]] = []  # (version, start_index)
    for i, line in enumerate(lines):
        m = SECTION_HEADER_RE.match(line)
        if m:
            sections.append((m.group("version"), i))
    if not sections:
        raise ValueError("No versioned sections found in CHANGELOG")
    # latest is the first section in the file after the title
    latest_version, start = sections[0]
    # find next section start or end of file
    end = len(lines)
    for _, idx in sections[1:]:
        if idx > start:
            end = idx
            break
    section_text = "\n".join(lines[start:end]).strip()
    return latest_version, section_text



def extract_latest_changelog_from_file(path: Path = CHANGELOG_PATH) -> Tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    return extract_latest_changelog(text)



def ensure_gh_cli() -> None:
    if subprocess.run(["which", "gh"], stdout=subprocess.DEVNULL).returncode != 0:
        raise RuntimeError("GitHub CLI (gh) is not installed. See https://cli.github.com/")



def main() -> int:
    print("🚀 Starting borgllm publish process...")

    print("🔍 Checking for uncommitted changes...")
    clean, status = git_is_clean()
    if not clean:
        print("❌ Error: Git working directory is dirty. Please commit or stash your changes before publishing.")
        if status:
            print(status)
        return 1
    print("✅ Git working directory is clean.")

    version = get_version_from_pyproject()
    if not version:
        print("❌ Error: Could not extract version from pyproject.toml")
        return 1
    print(f"🔖 Detected version: v{version}")

    tag = f"v{version}"
    print(f"🏷️ Checking for git tag {tag}...")
    if not git_tag_exists(tag):
        print(f"❌ Error: Git tag {tag} does not exist. Please create it before publishing.")
        print(f"Example: git tag {tag} && git push origin tag {tag}")
        return 1
    print(f"✅ Git tag {tag} exists.")

    print(f"🚀 Creating GitHub Release {tag}...")
    ensure_gh_cli()

    # Extract latest changelog for release notes
    try:
        latest_version, section = extract_latest_changelog_from_file()
    except Exception as e:
        print(f"⚠️ Warning: Failed to extract latest changelog section: {e}")
        latest_version, section = version, ""

    # If versions mismatch, still proceed but warn
    if latest_version != version:
        print(f"⚠️ Warning: Latest changelog version {latest_version} does not match pyproject version {version}")

    # Write notes to a temp file to pass to gh
    import tempfile
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
        tf.write(section)
        notes_path = tf.name

    run(["gh", "release", "create", tag, "--title", f"borgllm v{version}", "--notes-file", notes_path])

    print("🎉 borgllm publish process completed successfully!")
    print(f"✨ GitHub Release {tag} created successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
