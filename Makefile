.PHONY: lint, run
lint:
	poetry run isort ./src/train.py
	poetry run flake8 ./src/train.py
	poetry run mypy ./src/train.py
test:
	poetry run pytest