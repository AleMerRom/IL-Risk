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
	@printf "\\documentclass[11pt]{article}\n" > $(SELECTED_MAIN).tex
	@printf "\\usepackage[a4paper,margin=1in]{geometry}\n" >> $(SELECTED_MAIN).tex
	@printf "\\usepackage[T1]{fontenc}\n" >> $(SELECTED_MAIN).tex
	@printf "\\usepackage[utf8]{inputenc}\n" >> $(SELECTED_MAIN).tex
	@printf "\\usepackage{lmodern}\n" >> $(SELECTED_MAIN).tex
	@printf "\\usepackage{amsmath,amssymb}\n" >> $(SELECTED_MAIN).tex
	@printf "\\usepackage{graphicx}\n" >> $(SELECTED_MAIN).tex
	@printf "\\usepackage{booktabs}\n" >> $(SELECTED_MAIN).tex
	@printf "\\usepackage{hyperref}\n\n" >> $(SELECTED_MAIN).tex
	@printf "\\title{Final Project Report (Selected Modules)}\n" >> $(SELECTED_MAIN).tex
	@printf "\\author{}\n" >> $(SELECTED_MAIN).tex
	@printf "\\date{\\today}\n\n" >> $(SELECTED_MAIN).tex
	@printf "\\begin{document}\n\\maketitle\n\\tableofcontents\n\\newpage\n\n" >> $(SELECTED_MAIN).tex
	@printf "\\section{Introduction}\n%% Optional brief intro for selected build.\n\n" >> $(SELECTED_MAIN).tex
	@for m in $(MODULES); do printf "\\input{$$m.tex}\n" >> $(SELECTED_MAIN).tex; done
	@printf "\n\\section{Conclusion}\n%% Optional brief conclusion for selected build.\n\n\\end{document}\n" >> $(SELECTED_MAIN).tex

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
