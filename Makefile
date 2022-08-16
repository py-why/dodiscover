build-docs:
	@echo "Building documentation"
	make -C doc/ clean
	make -C doc/ html-noplot
	cd doc/ && make view