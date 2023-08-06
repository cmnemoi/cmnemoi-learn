install:
	poetry install

lint:
	poetry run black .
	poetry run mypy .
	poetry run pylint cmnemoi_learn tests

test:
	poetry run pytest -v --cov=cmnemoi_learn/