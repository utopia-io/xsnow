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
	sphinx-apidoc -f -e -o api/_generated ../src/xsnow

html: api
	sed -i '1s/^xsnow\.extensions\.//' api/_generated/*.rst  # make extension titles more legible
	sed -i '1s/^xsnow\.//' api/_generated/*.rst  # make module titles more legible
	pip install -e ..
	jupyter-book build $(SOURCEDIR) --path-output $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR) api/_generated
