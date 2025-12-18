import os
import shutil
from pathlib import Path

import pytest
import yaml

from synthla_edu_v2.run import run


def test_smoke_quick(tmp_path, monkeypatch):
    # Use the provided quick config but set a short out_dir into tmp
    cfg = Path("configs/quick.yaml")
    assert cfg.exists(), "configs/quick.yaml must exist for smoke test"

    # Ensure data/raw paths exist and contain minimal CSVs for functions that will be called
    # We'll create tiny CSVs expected by OULAD and ASSISTments data builders to avoid heavy deps.
    raw_dir_oulad = tmp_path / "data" / "raw" / "oulad"
    raw_dir_oulad.mkdir(parents=True)

    raw_dir_assist = tmp_path / "data" / "raw" / "assistments"
    raw_dir_assist.mkdir(parents=True)

    # OULAD files — create multiple rows with required columns for split and aggregations
    n = 50
    students = [str(i) for i in range(1, n + 1)]
    (raw_dir_oulad / "studentInfo.csv").write_text("code_module,code_presentation,id_student,final_result\n" + "\n".join([f"M1,2013J,{s},{'Withdrawn' if int(s) % 2 == 0 else 'Pass'}" for s in students]))
    (raw_dir_oulad / "studentRegistration.csv").write_text("code_module,code_presentation,id_student,date_registration,date_unregistration\n" + "\n".join([f"M1,2013J,{s},1,\"\"" for s in students]))
    (raw_dir_oulad / "studentVle.csv").write_text("code_module,code_presentation,id_student,sum_click,id_site,date\n" + "\n".join([f"M1,2013J,{s},1,1,2020-01-01" for s in students]))
    (raw_dir_oulad / "studentAssessment.csv").write_text("code_module,code_presentation,id_student,id_assessment,score\n" + "\n".join([f"M1,2013J,{s},10,80" for s in students]))
    (raw_dir_oulad / "assessments.csv").write_text("code_module,code_presentation,id_assessment,weight\nM1,2013J,10,1\n")
    (raw_dir_oulad / "vle.csv").write_text("code_module,code_presentation,id_student,id_site\n" + "\n".join([f"M1,2013J,{s},1" for s in students]))

    # ASSISTments minimal files with required columns
    assist_cols = [
        "user_id","problem_id","skill_id","original","attempt_count","ms_first_response","tutor_mode","answer_type","type","hint_count","hint_total","overlap_time","template_id","first_action","bottom_hint","opportunity","opportunity_original","position","correct"
    ]
    assist_rows = "\n".join([f"{i},1,1,1,1,100,hint,answer,type,0,0,0,0,first,0,1,1,1,1" for i in range(1, 20)])
    (raw_dir_assist / "assistments_2009_2010.csv").write_text(
        ",".join(assist_cols) + "\n" + assist_rows + "\n"
    )

    # Load YAML, modify raw paths and out_dir, and write a temporary config
    cfg_dict = yaml.safe_load(cfg.read_text())
    for ds in cfg_dict["datasets"]:
        name = str(ds.get("name", "")).lower()
        if "oulad" in name:
            ds["raw_path"] = str(raw_dir_oulad)
        elif "assist" in name:
            ds["raw_path"] = str(raw_dir_assist)
        else:
            ds["raw_path"] = str(tmp_path / "data" / "raw")
    cfg_dict["out_dir"] = str(tmp_path / "runs")

    tmp_cfg = tmp_path / "cfg.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg_dict))

    out = run(tmp_cfg)
    assert (Path(out) / "oulad" / "real_train.parquet").exists() or (Path(out) / "assistments" / "real_train.parquet").exists()
