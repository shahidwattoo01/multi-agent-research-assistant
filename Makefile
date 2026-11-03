.PHONY: install test lint format run serve docker

install:
	pip install -e ".[dev,api,autogen]"

test:
	pytest -q

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests
	ruff check --fix src tests

run:
	research-assistant run "$(Q)"

serve:
	research-assistant serve

docker:
	docker build -t research-assistant .
