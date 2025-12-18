"""
Sample data loader for SYNTHLA-EDU V2.

This module provides example functions to load publicly available
educational datasets and prepare them for the SYNTHLA-EDU V2 pipeline.

Datasets:
  - OULAD: Open University Learning Analytics Dataset
  - ASSISTments: ASSISTments 2009-2010 interaction data
"""

from pathlib import Path
import urllib.request
import zipfile
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def download_oulad_sample(
    target_dir: str | Path = "data/raw/oulad",
    courses: list[str] | None = None,
) -> Path:
    """
    Download a sample of OULAD data from the official repository.
    
    Note: Full OULAD dataset requires manual download from:
    https://analyse.kmi.open.ac.uk/open_dataset
    
    This function downloads a minimal sample for demonstration.
    
    Args:
        target_dir: Where to save the data
        courses: List of course codes to download (default: ['AAA', 'BBB', 'CCC'])
    
    Returns:
        Path to the downloaded data directory
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating sample OULAD data at {target_dir}")
    
    if courses is None:
        courses = ['AAA', 'BBB', 'CCC']
    
    # Create sample OULAD files with required columns
    n_students = 100
    
    # 1. studentInfo.csv
    student_info = []
    for course in courses:
        for student_id in range(1, n_students + 1):
            student_info.append({
                'code_module': course,
                'code_presentation': '2013J',
                'id_student': student_id,
                'final_result': 'Pass' if student_id % 2 == 0 else 'Withdrawn'
            })
    pd.DataFrame(student_info).to_csv(target_dir / "studentInfo.csv", index=False)
    
    # 2. studentRegistration.csv
    student_reg = []
    for course in courses:
        for student_id in range(1, n_students + 1):
            student_reg.append({
                'code_module': course,
                'code_presentation': '2013J',
                'id_student': student_id,
                'date_registration': 1,
                'date_unregistration': 200
            })
    pd.DataFrame(student_reg).to_csv(target_dir / "studentRegistration.csv", index=False)
    
    # 3. studentVle.csv
    student_vle = []
    for course in courses:
        for student_id in range(1, n_students + 1):
            for day in range(1, 50):
                student_vle.append({
                    'code_module': course,
                    'code_presentation': '2013J',
                    'id_student': student_id,
                    'id_site': (day % 10) + 1,
                    'date': day,
                    'sum_click': day % 5
                })
    pd.DataFrame(student_vle).to_csv(target_dir / "studentVle.csv", index=False)
    
    # 4. studentAssessment.csv
    student_assess = []
    for course in courses:
        for student_id in range(1, n_students + 1):
            for assess_id in range(1, 5):
                student_assess.append({
                    'code_module': course,
                    'code_presentation': '2013J',
                    'id_student': student_id,
                    'id_assessment': assess_id,
                    'score': (student_id + assess_id) % 100
                })
    pd.DataFrame(student_assess).to_csv(target_dir / "studentAssessment.csv", index=False)
    
    # 5. assessments.csv
    assessments = []
    for course in courses:
        for assess_id in range(1, 5):
            assessments.append({
                'code_module': course,
                'code_presentation': '2013J',
                'id_assessment': assess_id,
                'assessment_type': 'Type' + str(assess_id % 3),
                'date': 100 + (assess_id * 20),
                'weight': 25
            })
    pd.DataFrame(assessments).to_csv(target_dir / "assessments.csv", index=False)
    
    # 6. vle.csv
    vle = []
    for course in courses:
        for site_id in range(1, 11):
            vle.append({
                'code_module': course,
                'code_presentation': '2013J',
                'id_site': site_id,
                'activity_type': 'Resource' if site_id % 2 == 0 else 'Forumng'
            })
    pd.DataFrame(vle).to_csv(target_dir / "vle.csv", index=False)
    
    logger.info(f"Created complete sample OULAD data at {target_dir}")
    logger.info(f"  - {len(student_info)} student records")
    logger.info(f"  - {len(student_vle)} VLE interactions")
    logger.info(f"  - {len(student_assess)} assessment submissions")
    return target_dir


def download_assistments_sample(
    target_dir: str | Path = "data/raw/assistments",
) -> Path:
    """
    Download a sample of ASSISTments 2009-2010 data.
    
    Note: Full ASSISTments dataset can be obtained from:
    https://www.assistments.org/
    
    This function creates a minimal sample for testing.
    
    Args:
        target_dir: Where to save the data
    
    Returns:
        Path to the downloaded data directory
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"ASSISTments data can be obtained from:")
    logger.info(f"https://www.assistments.org/")
    logger.info(f"Extract to: {target_dir}")
    
    # Create minimal sample with required columns
    assist_cols = [
        "user_id", "problem_id", "skill_id", "original", "attempt_count",
        "ms_first_response", "tutor_mode", "answer_type", "type",
        "hint_count", "hint_total", "overlap_time", "template_id",
        "first_action", "bottom_hint", "opportunity", "opportunity_original",
        "position", "correct"
    ]
    
    assist_rows = []
    for interaction_id in range(1, 1001):
        assist_rows.append({
            col: interaction_id % 10 if col != "correct" else interaction_id % 2
            for col in assist_cols
        })
    
    assist_df = pd.DataFrame(assist_rows)
    (target_dir / "assistments_2009_2010.csv").write_text(
        assist_df.to_csv(index=False)
    )
    
    logger.info(f"Created sample ASSISTments data at {target_dir}")
    return target_dir


def verify_dataset(
    dataset_name: str,
    raw_path: str | Path,
) -> dict[str, str | int]:
    """
    Verify that a dataset has all required files and columns.
    
    Args:
        dataset_name: 'oulad' or 'assistments'
        raw_path: Path to raw data directory
    
    Returns:
        Dictionary with verification results
    """
    raw_path = Path(raw_path)
    
    if dataset_name.lower() == "oulad":
        required_files = [
            "studentInfo.csv",
            "studentRegistration.csv",
            "studentVle.csv",
            "studentAssessment.csv",
            "assessments.csv",
            "vle.csv",
        ]
        required_cols = {
            "studentInfo.csv": ["code_module", "code_presentation", "id_student", "final_result"],
            "studentRegistration.csv": ["code_module", "code_presentation", "id_student", "date_registration"],
            "studentVle.csv": ["code_module", "code_presentation", "id_student", "sum_click", "date"],
            "studentAssessment.csv": ["code_module", "code_presentation", "id_student", "id_assessment", "score"],
            "assessments.csv": ["code_module", "code_presentation", "id_assessment"],
            "vle.csv": ["code_module", "code_presentation", "id_site"],
        }
    elif dataset_name.lower() == "assistments":
        required_files = ["assistments_2009_2010.csv"]
        required_cols = {
            "assistments_2009_2010.csv": [
                "user_id", "problem_id", "skill_id", "original", "attempt_count",
                "ms_first_response", "tutor_mode", "answer_type", "type",
                "hint_count", "hint_total", "overlap_time", "template_id",
                "first_action", "bottom_hint", "opportunity", "opportunity_original",
                "position", "correct"
            ]
        }
    else:
        return {"error": f"Unknown dataset: {dataset_name}"}
    
    results = {
        "dataset": dataset_name,
        "path": str(raw_path),
        "files_found": 0,
        "files_missing": [],
        "columns_ok": True,
        "issues": [],
    }
    
    for file_name in required_files:
        file_path = raw_path / file_name
        if file_path.exists():
            results["files_found"] += 1
            try:
                df = pd.read_csv(file_path)
                missing_cols = [c for c in required_cols[file_name] if c not in df.columns]
                if missing_cols:
                    results["columns_ok"] = False
                    results["issues"].append(f"{file_name}: missing columns {missing_cols}")
            except Exception as e:
                results["issues"].append(f"{file_name}: {str(e)}")
        else:
            results["files_missing"].append(file_name)
    
    results["status"] = "OK" if not results["issues"] else "FAIL"
    return results


if __name__ == "__main__":
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    # Download samples
    print("=" * 60)
    print("SYNTHLA-EDU V2: Sample Data Setup")
    print("=" * 60)
    
    # OULAD
    print("\n[1/2] Downloading OULAD sample...")
    oulad_path = download_oulad_sample()
    oulad_check = verify_dataset("oulad", oulad_path)
    print(json.dumps(oulad_check, indent=2))
    
    # ASSISTments
    print("\n[2/2] Downloading ASSISTments sample...")
    assist_path = download_assistments_sample()
    assist_check = verify_dataset("assistments", assist_path)
    print(json.dumps(assist_check, indent=2))
    
    print("\n" + "=" * 60)
    print("Setup complete! Data ready at:")
    print(f"  - {oulad_path}")
    print(f"  - {assist_path}")
    print("=" * 60)
