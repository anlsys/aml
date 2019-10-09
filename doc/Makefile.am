DATA_INSTALL_DIR=$(datadir)/aml
DOXYGEN_BUILD_DIR=./build-doxygen
SPHINX_BUILD_DIR=./build-sphinx

if BUILD_DOCS
build-docs:
	$(DOXYGEN) aml.doxy
	$(SPHINXBUILD) -v -W -b html -a . $(SPHINX_BUILD_DIR)

dist-hook: build-docs
	cp -r $(SPHINX_BUILD_DIR) $(distdir)
else
build-docs:
	@echo "not building documentation"

dist-hook: build-docs
endif

create-doc-dir:
	mkdir -p $(DATA_INSTALL_DIR)

install-doc-html: build-docs
	if [ -d $(SPHINX_BUILD_DIR) ]; then \
	cp -r $(SPHINX_BUILD_DIR) $(DATA_INSTALL_DIR)/html; \
	fi

install-data-local: create-doc-dir install-doc-html
uninstall-local:
	rm -rf $(DATA_INSTALL_DIR)

clean-local:
	rm -rf $(DOXYGEN_BUILD_DIR) $(SPHINX_BUILD_DIR)

EXTRA_DIST= aml.doxy conf.py pages index.rst img
