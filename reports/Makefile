TEX_MAIN := main
PDF := $(TEX_MAIN).pdf
DEFAULT_MODULES := module1 module2 module3 module4 module5
MODULES ?= $(DEFAULT_MODULES)
SELECTED_MAIN := main_selected
SELECTED_PDF := $(SELECTED_MAIN).pdf

.PHONY: all selected clean distclean open open-selected help

all: $(PDF)

$(PDF): $(TEX_MAIN).tex $(addsuffix .tex,$(DEFAULT_MODULES))
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX_MAIN).tex

selected: $(SELECTED_PDF)

$(SELECTED_MAIN).tex: $(addsuffix .tex,$(MODULES))
	@printf '%s\n' '\documentclass[11pt]{article}' > $(SELECTED_MAIN).tex
	@printf '%s\n' '\usepackage[a4paper,margin=1in]{geometry}' >> $(SELECTED_MAIN).tex
	@printf '%s\n' '\usepackage[T1]{fontenc}' >> $(SELECTED_MAIN).tex
	@printf '%s\n' '\usepackage[utf8]{inputenc}' >> $(SELECTED_MAIN).tex
	@printf '%s\n' '\usepackage{lmodern}' >> $(SELECTED_MAIN).tex
	@printf '%s\n' '\usepackage{amsmath,amssymb}' >> $(SELECTED_MAIN).tex
	@printf '%s\n' '\usepackage{graphicx}' >> $(SELECTED_MAIN).tex
	@printf '%s\n' '\usepackage{booktabs}' >> $(SELECTED_MAIN).tex
	@printf '%s\n\n' '\usepackage{hyperref}' >> $(SELECTED_MAIN).tex
	@printf '%s\n' '\title{Final Project Report (Selected Modules)}' >> $(SELECTED_MAIN).tex
	@printf '%s\n' '\author{}' >> $(SELECTED_MAIN).tex
	@printf '%s\n\n' '\date{\today}' >> $(SELECTED_MAIN).tex
	@printf '%s\n' '\begin{document}' >> $(SELECTED_MAIN).tex
	@printf '%s\n' '\maketitle' >> $(SELECTED_MAIN).tex
	@printf '%s\n' '\tableofcontents' >> $(SELECTED_MAIN).tex
	@printf '%s\n\n' '\newpage' >> $(SELECTED_MAIN).tex
	@printf '%s\n' '\section{Introduction}' >> $(SELECTED_MAIN).tex
	@printf '%s\n\n' '% Optional brief intro for selected build.' >> $(SELECTED_MAIN).tex
	@for m in $(MODULES); do printf '%s\n' "\\input{$$m.tex}" >> $(SELECTED_MAIN).tex; done
	@printf '%s\n' '' >> $(SELECTED_MAIN).tex
	@printf '%s\n' '\section{Conclusion}' >> $(SELECTED_MAIN).tex
	@printf '%s\n\n' '% Optional brief conclusion for selected build.' >> $(SELECTED_MAIN).tex
	@printf '%s\n' '\end{document}' >> $(SELECTED_MAIN).tex

$(SELECTED_PDF): $(SELECTED_MAIN).tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(SELECTED_MAIN).tex

clean:
	latexmk -c $(TEX_MAIN).tex
	latexmk -c $(SELECTED_MAIN).tex 2>/dev/null || true

distclean:
	latexmk -C $(TEX_MAIN).tex
	latexmk -C $(SELECTED_MAIN).tex 2>/dev/null || true
	rm -f $(SELECTED_MAIN).tex
	rm -f $(PDF)
	rm -f $(SELECTED_PDF)

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
	@echo "make clean           Remove aux files"
	@echo "make distclean       Remove aux files and generated PDFs"
