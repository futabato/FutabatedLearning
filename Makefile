.PHONY: lint, run
lint:
	poetry run isort ./src/
	poetry run black ./src/
	poetry run pflake8 ./src/
	poetry run mypy ./src/
test:
	poetry run pytest