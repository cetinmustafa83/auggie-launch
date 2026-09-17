# auggie-launch — developer tasks
# Run `make` or `make help` for the list.

PYTHON ?= python3
LAUNCH ?= auggie-launch

.PHONY: help check lint types test doctor clean install

help:
	@echo "auggie-launch"
	@echo ""
	@echo "  make check     lint + types + tests (the CI gate)"
	@echo "  make lint      ruff"
	@echo "  make types     mypy"
	@echo "  make test      unit tests"
	@echo "  make doctor    end-to-end diagnostics against the live upstream"
	@echo "  make install   install the launcher into ~/.local/bin"
	@echo "  make clean     remove caches"

check: lint types test
	@echo ""
	@echo "check passed"

lint:
	@$(PYTHON) -m ruff check auggie_launch/ test_modern_proxy.py

types:
	@$(PYTHON) -m mypy auggie_launch/

test:
	@$(PYTHON) -m unittest test_modern_proxy

doctor:
	@$(LAUNCH) --doctor

install:
	@./install.sh --skip-9router

clean:
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .ruff_cache .mypy_cache .pytest_cache
	@echo "cleaned"
