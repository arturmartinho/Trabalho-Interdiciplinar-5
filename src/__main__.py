from .calculadora import somar, subtrair


def main() -> None:
    print("Projeto Python modular")
    print("2 + 3 =", somar(2, 3))
    print("10 - 4 =", subtrair(10, 4))


if __name__ == "__main__":
    main()
