.PHONY: format test

format:
	ruff format .
	npx markdownlint-cli --fix .

test:
	python -m pytest tests/ -v
