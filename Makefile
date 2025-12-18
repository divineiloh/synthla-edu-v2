.PHONY: setup run_full run_quick test

setup:
	python -m pip install -U pip
	python -m pip install -r requirements.txt

run_full:
	python -m synthla_edu_v2.run --config configs/full.yaml

run_quick:
	python -m synthla_edu_v2.run --config configs/quick.yaml

test:
	python -m pytest -q
