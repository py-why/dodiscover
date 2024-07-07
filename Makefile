# simple makefile to simplify repetitive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
PYTESTS ?= pytest
CTAGS ?= ctags
CODESPELL_SKIPS ?= "*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii,*.pyc,*.doctree,*.pickle,*.inv,*.png,*.edf,*.touch,*.thickness,*.nofix,*.volume,*.defect_borders,*.mgh,lh.*,rh.*,COR-*,FreeSurferColorLUT.txt,*.examples,.xdebug_mris_calc,bad.segments,BadChannels,*.hist,empty_file,*.orig,*.js,*.map,*.ipynb,searchindex.dat,plot_*.rst,*.rst.txt,*.html,gdf_encodes.txt,dodiscover/_lib/*,doc/auto_examples/*"
CODESPELL_DIRS ?= dodiscover/ doc/ examples/ benchmarks/
all: clean inplace test test-doc

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-build:
	rm -rf build
	rm -rf dist

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean: clean-build clean-pyc clean-cache

pytest: test

test: in
	rm -f .coverage
	$(PYTESTS) dodiscover

test-doc: sample_data testing_data
	$(PYTESTS) --doctest-modules --doctest-ignore-import-errors --doctest-glob='*.rst' ./doc/

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count dodiscover examples; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

black:
	@if command -v black > /dev/null; then \
		echo "Running black"; \
		black dodiscover examples; \
	else \
		echo "black not found, please install it!"; \
		exit 1; \
	fi;
	@echo "black passed"

isort:
	@if command -v isort > /dev/null; then \
		echo "Running isort"; \
		isort dodiscover examples doc; \
	else \
		echo "isort not found, please install it!"; \
		exit 1; \
	fi;
	@echo "isort passed"

codespell:  # running manually
	@codespell -w -i 3 -q 3 -S $(CODESPELL_SKIPS) --ignore-words=.codespellignore $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@codespell -i 0 -q 7 -S $(CODESPELL_SKIPS) --ignore-words=.codespellignore $(CODESPELL_DIRS)

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle mne

docstyle: pydocstyle

build-doc:
	@echo "Building documentation"
	make -C doc/ clean
	make -C doc/ html
	cd doc/ && make view

build-doc-noplot:
	@echo "Building documentation"
	make -C doc/ clean
	make -C doc/ html-noplot
	cd doc/ && make view

run-checks:
	isort --check .
	black --check dodiscover examples
	mypy ./dodiscover
	@$(MAKE) pydocstyle
	@$(MAKE) codespell-error
	ruff .
	toml-sort ./pyproject.toml --check
	yamllint . -c .yamllint.yml --strict

pre-commit:
	@pre-commit run -a