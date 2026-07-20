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
	sed -i '1s/^xsnow\.io\./io./' api/_generated/*.rst  # keep I/O module titles under xsnow readable
	sed -i '1s/^xsnow\.//' api/_generated/*.rst  # make module titles more legible
	sed -i '/^Submodules$$/,/^Module contents$$/ { /^Module contents$$/!d; }' api/_generated/xsnow.io.rst  # avoid duplicate I/O submodule nav
	sed -i '/^Module contents$$/,+2d' api/_generated/xsnow.io.rst  # remove redundant generated section heading
	pip install -e "..[viz]"
	python ../dist/update_footer_citation.py
	jupyter-book build $(SOURCEDIR) --path-output $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR) api/_generated
