import faiss
from imsim.search import embed_and_search
import typer
import typing as t
import pathlib
import pprint
from dataclasses import dataclass, asdict

app = typer.Typer()

@dataclass
class Similarity:
    distance: float
    id: str | int

@dataclass
class Result:
    image_id: str | int
    similarities: list[Similarity]


def convert_to_dto(tags: pathlib.Path | None, images, results):

    if tags:
        with open(tags) as fp:
            tags_lines = [s.strip() for s in fp.readlines()]

        results = [
            Result(
                image_id=str(im),
                similarities=[
                    Similarity(distance=dd, id=tags_lines[ii])
                    for dd, ii in zip(d, i)
                ],
            )
            for im, (d, i) in zip(images, results)
        ]
        
    else:
        results = [
            Result(
                image_id=image_id,
                similarities=[
                    Similarity(distance=distance, id=id)
                    for distance, id in zip(d, i)
                ],
            )
            for image_id, (d, i) in enumerate(results)
        ]

    return results

@app.command("images")
def search_images(
    root_dir: t.Annotated[pathlib.Path, typer.Option()],
    model: t.Annotated[str, typer.Option()],
    index: t.Annotated[pathlib.Path, typer.Option()],
    top_n: t.Annotated[int, typer.Option()] = 3,
    tags: t.Annotated[t.Optional[pathlib.Path], typer.Option()] = None,
):
    images = sorted(pathlib.Path(root_dir).glob("**/*.jpg"))
    results = embed_and_search(faiss.read_index(str(index.resolve())), model, images, top_n)
    results = convert_to_dto(tags, images, results)
    pprint.pprint(results)

@app.command("image")
def search_image(
    image: t.Annotated[pathlib.Path, typer.Option()],
    model: t.Annotated[str, typer.Option()],
    index: t.Annotated[pathlib.Path, typer.Option()],
    top_n: t.Annotated[int, typer.Option()] = 3,
    tags: t.Annotated[t.Optional[pathlib.Path], typer.Option()] = None,
):
    results = embed_and_search(faiss.read_index(str(index.resolve())), model, [image], top_n)
    results = convert_to_dto(tags, [image], results)
    pprint.pprint([asdict(r) for r in results])

if __name__ == "__main__":
    app()
