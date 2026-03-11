# Wind-Vision Makefile

.PHONY: setup fetch extract train eval predict serve test clean

setup:
	python -m venv .venv
	./.venv/bin/pip install -e ".[http2]"
	playwright install chromium

fetch:
	python -m wind_vision.cli fetch

extract:
	python -m wind_vision.cli extract

train:
	python -m wind_vision.cli train

eval:
	python -m wind_vision.cli eval

predict:
	@python -m wind_vision.cli predict $(IMG)

serve:
	uvicorn src.wind_vision.api.server:app --reload

test:
	pytest tests/ -v

docker:
	docker build -t wind-vision .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info
