# convenience recipe for poetry to call to build the docs from the root
# directory
build-docs:
	@echo "Building documentation"
	make -C doc/ clean
	make -C doc/ html-noplot
	cd doc/ && make view
