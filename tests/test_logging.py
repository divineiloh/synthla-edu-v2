from pathlib import Path

from synthla_edu_v2.run import run


def test_run_writes_log(tmp_path):
    cfg = Path("configs/quick.yaml")
    # modify into tmp config as in smoke test
    import yaml

    cfg_dict = yaml.safe_load(cfg.read_text())
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    n = 40
    students = [str(i) for i in range(1, n + 1)]
    (raw_dir / "studentInfo.csv").write_text("code_module,code_presentation,id_student,final_result\n" + "\n".join([f"M1,2013J,{s},Withdrawn" for s in students]))
    (raw_dir / "studentRegistration.csv").write_text("code_module,code_presentation,id_student,date_registration,date_unregistration\n" + "\n".join([f"M1,2013J,{s},1,\"\"" for s in students]))
    raw_dir_oulad = tmp_path / "data" / "raw" / "oulad"
    raw_dir_oulad.mkdir(parents=True)

    raw_dir_assist = tmp_path / "data" / "raw" / "assistments"
    raw_dir_assist.mkdir(parents=True)

    (raw_dir_oulad / "studentInfo.csv").write_text("code_module,code_presentation,id_student,final_result\n" + "\n".join([f"M1,2013J,{s},Withdrawn" for s in students]))
    (raw_dir_oulad / "studentRegistration.csv").write_text("code_module,code_presentation,id_student,date_registration,date_unregistration\n" + "\n".join([f"M1,2013J,{s},1,\"\"" for s in students]))
    (raw_dir_oulad / "studentVle.csv").write_text("code_module,code_presentation,id_student,sum_click,id_site,date\n" + "\n".join([f"M1,2013J,{s},1,1,2020-01-01" for s in students]))
    (raw_dir_oulad / "studentAssessment.csv").write_text("code_module,code_presentation,id_student,id_assessment,score\n" + "\n".join([f"M1,2013J,{s},10,80" for s in students]))
    (raw_dir_oulad / "assessments.csv").write_text("code_module,code_presentation,id_assessment,weight\nM1,2013J,10,1\n")
    (raw_dir_oulad / "vle.csv").write_text("code_module,code_presentation,id_student,id_site\n" + "\n".join([f"M1,2013J,{s},1" for s in students]))

    assist_cols = [
        "user_id","problem_id","skill_id","original","attempt_count","ms_first_response","tutor_mode","answer_type","type","hint_count","hint_total","overlap_time","template_id","first_action","bottom_hint","opportunity","opportunity_original","position","correct"
    ]
    assist_rows = "\n".join([f"{i},1,1,1,1,100,hint,answer,type,0,0,0,0,first,0,1,1,1,1" for i in range(1, 20)])
    (raw_dir_assist / "assistments_2009_2010.csv").write_text(
        ",".join(assist_cols) + "\n" + assist_rows + "\n"
    )

    for ds in cfg_dict["datasets"]:
        if str(ds["name"]).lower() == "oulad":
            ds["raw_path"] = str(raw_dir_oulad)
        else:
            ds["raw_path"] = str(raw_dir_assist)


    for ds in cfg_dict["datasets"]:
        ds["raw_path"] = str(raw_dir)
    cfg_dict["out_dir"] = str(tmp_path / "runs")
    tmp_cfg = tmp_path / "cfg.yaml"
    tmp_cfg.write_text(yaml.safe_dump(cfg_dict))

    out = run(tmp_cfg)
    log = Path(out) / "run.log"
    assert log.exists(), "run.log should be created"
    txt = log.read_text()
    assert "Fitting synthesizer" in txt or "Sampling" in txt or "Computed SDMetrics" in txt
