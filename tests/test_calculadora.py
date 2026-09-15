from calculadora import somar, subtrair


def test_somar():
    assert somar(2, 3) == 5


def test_somar_numeros_negativos():
    assert somar(-2, -3) == -5


def test_subtrair():
    assert subtrair(10, 4) == 6
