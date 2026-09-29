import typer
from rich import print

app = typer.Typer(no_args_is_help=True)


@app.command()
def info() -> None:
    print("[bold]nebius-physical-ai[/bold] — Physical AI track entry")


@app.command()
def version() -> None:
    print("0.1.0")
