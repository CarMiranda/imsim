import pathlib
import faiss
from imsim.index import update_index
import typer
import typing as t

app = typer.Typer()


@app.command(name="create")
def create_index(
    output_path: t.Annotated[pathlib.Path, typer.Option()],
    model_name: t.Annotated[str, typer.Option()],
    root_dir: t.Annotated[pathlib.Path, typer.Option()],
):
    index = faiss.IndexFlatL2(384)
    images = sorted(pathlib.Path(root_dir).glob("**/*.jpg"))

    print("Creating index...")
    update_index(index, model_name, images)

    faiss.write_index(index, str(output_path.resolve()))
    with open(output_path.with_name(f"{output_path.stem}-names.txt"), "w") as fp:
        fp.write('\n'.join(map(str, images)))


@app.command(name="update")
def increase_index(
    input_path: t.Annotated[pathlib.Path, typer.Option()],
    output_path: t.Annotated[pathlib.Path, typer.Option()],
    model_name: t.Annotated[str, typer.Option()],
    root_dir: t.Annotated[pathlib.Path, typer.Option()],
):
    index = faiss.read_index(str(input_path.resolve()))
    images = sorted(pathlib.Path(root_dir).glob("**/*.jpg"))

    print("Updating index...")
    update_index(index, model_name, images)

    faiss.write_index(index, str(output_path.resolve()))


if __name__ == "__main__":
    app()
