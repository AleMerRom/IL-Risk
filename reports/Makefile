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
	@printf '%s\n' '\documentclass[11pt]{article}' > $(SELECTED_TEX)
	@printf '%s\n' '\usepackage[a4paper,margin=1in]{geometry}' >> $(SELECTED_TEX)
	@printf '%s\n' '\usepackage[T1]{fontenc}' >> $(SELECTED_TEX)
	@printf '%s\n' '\usepackage[utf8]{inputenc}' >> $(SELECTED_TEX)
	@printf '%s\n' '\usepackage{lmodern}' >> $(SELECTED_TEX)
	@printf '%s\n' '\usepackage{amsmath,amssymb}' >> $(SELECTED_TEX)
	@printf '%s\n' '\usepackage{graphicx}' >> $(SELECTED_TEX)
	@printf '%s\n' '\usepackage{booktabs}' >> $(SELECTED_TEX)
	@printf '%s\n' '\usepackage{hyperref}' >> $(SELECTED_TEX)
	@printf '%s\n\n' '\newcommand{\num}[1]{#1}' >> $(SELECTED_TEX)
	@printf '%s\n' '\title{Final Project Report (Selected Modules)}' >> $(SELECTED_TEX)
	@printf '%s\n' '\author{}' >> $(SELECTED_TEX)
	@printf '%s\n\n' '\date{\today}' >> $(SELECTED_TEX)
	@printf '%s\n' '\begin{document}' >> $(SELECTED_TEX)
	@printf '%s\n' '\maketitle' >> $(SELECTED_TEX)
	@printf '%s\n' '\tableofcontents' >> $(SELECTED_TEX)
	@printf '%s\n\n' '\newpage' >> $(SELECTED_TEX)
	@printf '%s\n' '\section{Introduction}' >> $(SELECTED_TEX)
	@printf '%s\n\n' '% Optional brief intro for selected build.' >> $(SELECTED_TEX)
	@for m in $(MODULES); do printf '%s\n' "\\input{../$$m.tex}" >> $(SELECTED_TEX); done
	@printf '%s\n' '' >> $(SELECTED_TEX)
	@printf '%s\n' '\section{Conclusion}' >> $(SELECTED_TEX)
	@printf '%s\n\n' '% Optional brief conclusion for selected build.' >> $(SELECTED_TEX)
	@printf '%s\n' '\end{document}' >> $(SELECTED_TEX)
	
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