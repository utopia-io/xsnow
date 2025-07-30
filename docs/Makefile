SPHINXOPTS    =
SPHINXBUILD   = jupyter-book
SOURCEDIR     = .
BUILDDIR      = _build/html

.PHONY: help deps api tutorials build clean

help:
	@echo "Targets:"
	@echo "  deps       Install documentation dependencies"
	@echo "  api        Auto-generate API stubs (sphinx-apidoc)"
	@echo "  tutorials  Build tutorials via Jupyter Book"
	@echo "  build      Build everything (api + tutorials)"
	@echo "  clean      Remove build outputs"

deps:
	pip install -r requirements.txt

api:
	@echo "Generating API docs…"
	# assumes your package is at ../your_package
	sphinx-apidoc -o api ../your_package

tutorials:
	@echo "Building tutorials & API site…"
	$(SPHINXBUILD) build .

build: api tutorials

clean:
	rm -rf _build
