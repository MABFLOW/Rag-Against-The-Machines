from .chuncker import Chuncker


if __name__ == "__main__":
    ch = Chuncker("src/test.py")
    ch.run()