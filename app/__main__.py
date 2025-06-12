import asyncio

import typer

app = typer.Typer()


@app.command()
def record():
    from app.core.record import main as main_record

    try:
        asyncio.run(main_record())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting.")


@app.command()
def process():
    from app.core.process import main

    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting.")


@app.command()
def train():
    from app.core.train import main

    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting.")


@app.command()
def predict():
    from app.core.predict import main

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting.")


if __name__ == "__main__":
    app()
