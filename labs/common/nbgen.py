"""Хелпер для сборки Jupyter-ноутбуков программно."""
import nbformat as nbf


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build(cells: list, path: str):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata.kernelspec = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    nb.metadata.language_info = {"name": "python", "version": "3.11.9"}
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Written: {path}")
