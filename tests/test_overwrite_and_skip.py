import yaml
from pathlib import Path

from synthla_edu_v2.run import run
from synthla_edu_v2.eval.utility import run_utility
import pandas as pd


def test_run_overwrites_out_dir(tmp_path):
    cfg = Path("configs/quick.yaml")
    cfg_dict = yaml.safe_load(cfg.read_text())
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    # Create minimal but valid OULAD files — with enough students for train/test split
    n_students = 20
    students = [str(i) for i in range(1, n_students + 1)]
    (raw_dir / "studentInfo.csv").write_text("code_module,code_presentation,id_student,final_result\n" + "\n".join([f"M1,2013J,{s},{'Pass' if int(s) % 2 == 0 else 'Withdrawn'}" for s in students]))
    (raw_dir / "studentRegistration.csv").write_text("code_module,code_presentation,id_student,date_registration,date_unregistration\n" + "\n".join([f"M1,2013J,{s},1," for s in students]))
    (raw_dir / "studentVle.csv").write_text("code_module,code_presentation,id_student,sum_click,id_site,date\n" + "\n".join([f"M1,2013J,{s},1,1,2020-01-01" for s in students]))
    (raw_dir / "studentAssessment.csv").write_text("code_module,code_presentation,id_student,id_assessment,score\n" + "\n".join([f"M1,2013J,{s},10,{80 + int(s) % 20}" for s in students]))
    (raw_dir / "assessments.csv").write_text("code_module,code_presentation,id_assessment,weight\nM1,2013J,10,1\n")
    (raw_dir / "vle.csv").write_text("code_module,code_presentation,id_student,id_site\n" + "\n".join([f"M1,2013J,{s},1" for s in students]))
    # create assistments in a small separate raw dir
    raw_dir_assist = tmp_path / "data" / "raw" / "assistments"
    raw_dir_assist.mkdir(parents=True)
    assist_cols = [
        "user_id","problem_id","skill_id","original","attempt_count","ms_first_response","tutor_mode","answer_type","type","hint_count","hint_total","overlap_time","template_id","first_action","bottom_hint","opportunity","opportunity_original","position","correct"
    ]
    # Create multiple assist rows to allow train/test split
    assist_rows = "\n".join([f"{i},1,1,1,1,100,hint,answer,type,0,0,0,0,first,0,1,1,1,{i % 2}" for i in range(1, 20)])
    (raw_dir_assist / "assistments_2009_2010.csv").write_text(
        ",".join(assist_cols) + "\n" + assist_rows + "\n"
    )

    for ds in cfg_dict["datasets"]:
        if str(ds["name"]).lower() == "oulad":
            ds["raw_path"] = str(raw_dir)
        else:
            ds["raw_path"] = str(raw_dir_assist)

    out = tmp_path / "runs"
    # place a marker that should be removed by run()
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "old_marker.txt"
    marker.write_text("old")

    cfg_dict["out_dir"] = str(out)
    tmp_cfg = tmp_path / "cfg.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg_dict))

    run(tmp_cfg)
    assert not marker.exists(), "Old files in out_dir should be removed at start of run"


def test_run_utility_skips_single_class():
    # Create synthetic train with single-class target
    syn = pd.DataFrame({"feature1": [1, 2, 3], "dropout": [1, 1, 1]})
    # Create real_test with both classes to test skipping due to train only
    real_test = pd.DataFrame({"feature1": [4, 5], "dropout": [1, 0]})

    res = run_utility(
        task="classification",
        target_col="dropout",
        id_cols=[],
        feature_drop_cols=[],
        model_names=["logreg"],
        syn_train=syn,
        real_test=real_test,
        random_state=0,
    )
    assert res.metrics == {}, "Metrics should be empty when training data has single class"
    assert res.predictions == {}, "Predictions should be empty when training data has single class"
