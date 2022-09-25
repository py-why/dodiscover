# convenience recipe for poetry to call to build the docs from the root
# directory
build-docs:
	@echo "Building documentation"
	make -C doc/ clean
	make -C doc/ html-noplot
	cd doc/ && make view

clean-pyc:
find . -name "*.pyc" | xargs rm -f

clean-so:
find . -name "*.so" | xargs rm -f
find . -name "*.pyd" | xargs rm -f

clean-build:
rm -rf _build build dist dodiscover.egg-info

clean-ctags:
rm -f tags

clean-cache:
find . -name "__pycache__" | xargs rm -rf

clean-test:
rm -rf .pytest_cache .mypy_cache .ipynb_checkpoints
rm junit-results.xml

clean: clean-build clean-pyc clean-so clean-ctags clean-cache clean-test