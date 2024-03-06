import typer
from imsim.cli import search, vectorize


app = typer.Typer()
app.add_typer(search.app, name="search")
app.add_typer(vectorize.app, name="vectorize")

if __name__ == "__main__":
    app()
