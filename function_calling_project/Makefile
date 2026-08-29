.PHONY: install run debug test integration clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

test:
	uv run pytest tests/ -v

integration:
	uv run python -m src
	uv run python tests/test_generator_full_batch.py

clean:
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".pytest_cache" \) -prune -exec rm -rf {} +
	rm -rf data/output

lint:
	uv run flake8 src/
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src/
	uv run mypy . --strict
