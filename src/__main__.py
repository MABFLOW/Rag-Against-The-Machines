from .engine import Engine
import fire


if __name__ == "__main__":
    try:
        fire.Fire(Engine)
    except Exception as e:
        print(e)
