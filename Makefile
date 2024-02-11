.PHONY: lint, run
lint:
	poetry run ruff format .
	poetry run ruff check . --fix
test:
	poetry run pytest ./tests/