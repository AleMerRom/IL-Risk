TEX_MAIN := main
OUT_DIR := out

PDF := $(OUT_DIR)/$(TEX_MAIN).pdf

DEFAULT_MODULES := module1 module2 module3 module4 module5
MODULES ?= $(DEFAULT_MODULES)

SELECTED_MAIN := main_selected
SELECTED_TEX := $(OUT_DIR)/$(SELECTED_MAIN).tex
SELECTED_PDF := $(OUT_DIR)/$(SELECTED_MAIN).pdf

LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=$(OUT_DIR)

.PHONY: all selected clean distclean open open-selected help

all: $(PDF)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(PDF): $(TEX_MAIN).tex $(addsuffix .tex,$(DEFAULT_MODULES)) | $(OUT_DIR)
	$(LATEXMK) $(TEX_MAIN).tex

selected: $(SELECTED_PDF)

$(SELECTED_TEX): $(addsuffix .tex,$(MODULES)) | $(OUT_DIR)
	@printf "\\documentclass[11pt]{article}\n" > $(SELECTED_TEX)
	@printf "\\usepackage[a4paper,margin=1in]{geometry}\n" >> $(SELECTED_TEX)
	@printf "\\usepackage[T1]{fontenc}\n" >> $(SELECTED_TEX)
	@printf "\\usepackage[utf8]{inputenc}\n" >> $(SELECTED_TEX)
	@printf "\\usepackage{lmodern}\n" >> $(SELECTED_TEX)
	@printf "\\usepackage{amsmath,amssymb}\n" >> $(SELECTED_TEX)
	@printf "\\usepackage{graphicx}\n" >> $(SELECTED_TEX)
	@printf "\\usepackage{booktabs}\n" >> $(SELECTED_TEX)
	@printf "\\usepackage{hyperref}\n\n" >> $(SELECTED_TEX)
	@printf "\\title{Final Project Report (Selected Modules)}\n" >> $(SELECTED_TEX)
	@printf "\\author{}\n" >> $(SELECTED_TEX)
	@printf "\\date{\\today}\n\n" >> $(SELECTED_TEX)
	@printf "\\begin{document}\n\\maketitle\n\\tableofcontents\n\\newpage\n\n" >> $(SELECTED_TEX)
	@printf "\\section{Introduction}\n%% Optional brief intro for selected build.\n\n" >> $(SELECTED_TEX)
	@for m in $(MODULES); do printf "\\input{../$$m.tex}\n" >> $(SELECTED_TEX); done
	@printf "\n\\section{Conclusion}\n%% Optional brief conclusion for selected build.\n\n\\end{document}\n" >> $(SELECTED_TEX)

$(SELECTED_PDF): $(SELECTED_TEX) | $(OUT_DIR)
	cd $(OUT_DIR) && latexmk -pdf -interaction=nonstopmode -halt-on-error $(SELECTED_MAIN).tex

clean:
	rm -rf $(OUT_DIR)

distclean:
	rm -rf $(OUT_DIR)

open: $(PDF)
	open $(PDF)

open-selected: $(SELECTED_PDF)
	open $(SELECTED_PDF)

help:
	@echo "make                 Build full report from main.tex"
	@echo "make selected        Build selected modules (default: all)"
	@echo "make selected MODULES=\"module3 module5\""
	@echo "make open            Open full report PDF"
	@echo "make open-selected MODULES=\"module2 module4\""
	@echo "make clean           Remove generated build files"
	@echo "make distclean       Remove generated build files"