DATA_INSTALL_DIR=$(datadir)/aml
DOXYGEN_BUILD_DIR=./build-doxygen
SPHINX_BUILD_DIR=./build-sphinx

doxygen:
	doxygen aml.doxy

create-doc-dir:
	mkdir -p $(DATA_INSTALL_DIR)

install-doc-html:
	if [ -d $(DOXYGEN_BUILD_DIR)/html ]; then \
	cp -r $(DOXYGEN_BUILD_DIR)/html $(DATA_INSTALL_DIR); \
	fi

install-data-local: create-doc-dir install-doc-html

uninstall-local:
	rm -rf $(DATA_INSTALL_DIR)

clean-local:
	rm -rf $(DOXYGEN_BUILD_DIR) $(SPHINX_BUILD_DIR)

dist-hook: doxygen
	cp -r $(DOXYGEN_BUILD_DIR) $(distdir)


