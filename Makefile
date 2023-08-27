.PHONY: lint, run
lint:
	poetry run isort ./src/ ./tests/
	poetry run black ./src/ ./tests/
	poetry run pflake8 ./src/ ./tests/
	poetry run mypy ./src/ ./tests/
test:
	poetry run pytest ./tests/