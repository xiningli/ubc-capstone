rm -r build
mkdir -p build
pdflatex -synctex=1 -shell-escape -interaction=nonstopmode --output-dir build report.tex
# pdflatex -synctex=1 -shell-escape -interaction=nonstopmode --output-dir build report.tex
