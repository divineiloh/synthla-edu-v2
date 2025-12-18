"""
Deployment Verification Script
Checks that all required files are present for GitHub deployment.
"""
import os
from pathlib import Path
from typing import List, Tuple


def check_files_exist(base_path: Path, required_files: List[str]) -> Tuple[List[str], List[str]]:
    """Check which required files exist and which are missing."""
    existing = []
    missing = []
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            existing.append(file_path)
        else:
            missing.append(file_path)
    
    return existing, missing


def main():
    base_path = Path(__file__).parent
    
    # Required source files
    required_files = [
        # Core
        "src/synthla_edu_v2/__init__.py",
        "src/synthla_edu_v2/config.py",
        "src/synthla_edu_v2/run.py",
        "src/synthla_edu_v2/utils.py",
        
        # Data loaders
        "src/synthla_edu_v2/data/__init__.py",
        "src/synthla_edu_v2/data/assistments.py",
        "src/synthla_edu_v2/data/oulad.py",
        "src/synthla_edu_v2/data/split.py",
        "src/synthla_edu_v2/data/sample_loader.py",
        
        # Evaluation
        "src/synthla_edu_v2/eval/__init__.py",
        "src/synthla_edu_v2/eval/c2st.py",
        "src/synthla_edu_v2/eval/mia.py",
        "src/synthla_edu_v2/eval/models.py",
        "src/synthla_edu_v2/eval/preprocess.py",
        "src/synthla_edu_v2/eval/quality.py",
        "src/synthla_edu_v2/eval/reporting.py",
        "src/synthla_edu_v2/eval/stats.py",
        "src/synthla_edu_v2/eval/utility.py",
        
        # Synthesizers
        "src/synthla_edu_v2/synth/__init__.py",
        "src/synthla_edu_v2/synth/base.py",
        "src/synthla_edu_v2/synth/sdv_wrappers.py",
        "src/synthla_edu_v2/synth/tabddpm_wrappers.py",
        
        # Tests
        "tests/test_config.py",
        "tests/test_data_loading.py",
        "tests/test_eval.py",
        "tests/test_e2e.py",
        "tests/test_overwrite_and_skip.py",
        "tests/test_synth.py",
        
        # Configs
        "configs/quick.yaml",
        "configs/full.yaml",
        "configs/minimal.yaml",
        
        # Docker & CI
        "Dockerfile",
        ".dockerignore",
        ".github/workflows/ci.yml",
        ".github/workflows/docker.yml",
        
        # Dependencies
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-locked.txt",
        "pyproject.toml",
        
        # Documentation
        "README.md",
        "USAGE.md",
        "QUICKREF.md",
        "DEPLOYMENT.md",
        
        # Build
        "Makefile",
        ".gitignore",
    ]
    
    existing, missing = check_files_exist(base_path, required_files)
    
    print("=" * 70)
    print("SYNTHLA-EDU V2 - Deployment Verification")
    print("=" * 70)
    print()
    
    print(f"✅ Existing files: {len(existing)}/{len(required_files)}")
    print(f"❌ Missing files: {len(missing)}/{len(required_files)}")
    print()
    
    if missing:
        print("Missing files:")
        for file_path in sorted(missing):
            print(f"  - {file_path}")
        print()
        return 1
    
    print("✅ All required files are present!")
    print()
    
    # Check for files that should NOT be committed
    print("Checking for files that should be excluded from Git...")
    excluded_patterns = [
        "data/raw",
        "data/processed",
        "runs",
        "__pycache__",
        ".pytest_cache",
        ".venv",
    ]
    
    warnings = []
    for pattern in excluded_patterns:
        path = base_path / pattern
        if path.exists():
            warnings.append(pattern)
    
    if warnings:
        print(f"⚠️  Found {len(warnings)} directories that should be in .gitignore:")
        for w in warnings:
            print(f"  - {w}/")
        print("  (These will be excluded by .gitignore)")
    else:
        print("✅ No excluded directories found")
    
    print()
    print("=" * 70)
    print("Deployment Status: READY ✅")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. git init (if not already initialized)")
    print("  2. git add .")
    print("  3. git commit -m 'Initial commit: SYNTHLA-EDU V2'")
    print("  4. git remote add origin <your-repo-url>")
    print("  5. git push -u origin main")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
