.PHONY: lint, run
lint:
	poetry run ruff check . --fix
	poetry run ruff format .
	poetry run mypy .
test:
	poetry run pytest ./tests/