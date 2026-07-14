from .engine import Engine


if __name__ == "__main__":
    ch = Engine("data_chunked.json")

    ch.run()
    