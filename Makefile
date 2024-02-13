.PHONY: lint, run
lint:
	poetry run ruff format .
	poetry run ruff check . --fix
	poetry run mypy .
test:
	poetry run pytest ./tests/