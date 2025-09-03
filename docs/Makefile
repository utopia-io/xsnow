SOURCEDIR = .
BUILDDIR  = _build

.PHONY: help deps api tutorials build clean

help:
	@echo "Targets:"
	@echo "  deps       Install documentation dependencies"
	@echo "  api        Auto-generate API stubs (sphinx-apidoc)"
	@echo "  html       Build HTML site (api + tutorials)"
	@echo "  clean      Remove build outputs"

deps:
	pip install -r requirements.txt

api:
	@echo "Generating API docs…"
	sphinx-apidoc -f -e -o api ../src/xsnow

html: api
	pip install -e ..
	jupyter-book build $(SOURCEDIR) --path-output $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR) api
