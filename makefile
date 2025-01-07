

# HELP: black: [Static Analysis] Run Black
.PHONY: black
black:
	@echo "🚀 Running Black..."
	black --check --diff --config pyproject.toml src/
	@echo -e $(GREEN)"Looks good! 🎉"$(NOCOLOR)

# HELP: isort: [Static Analysis] Run isort
.PHONY: isort
isort:
	@echo "🚀 Running isort..."
	isort src/  -c --settings-path pyproject.toml
	@echo -e $(GREEN)"Looks good! 🎉"$(NOCOLOR)

# HELP: format-code: [Static Analysis] Format code using available formatting tools
.PHONY: format-code
format-code:
	@echo "🚀 Formatting code..."
	isort src/ --settings-path pyproject.toml
	black --config pyproject.toml src/
	@echo -e $(GREEN)"Done! ✅"$(NOCOLOR)

