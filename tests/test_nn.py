import numpy as np

from tictactoe.nn import MLP, Adam


def test_gradientes_coinciden_con_numericos():
    net = MLP([5, 8, 6, 3], seed=1)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 5)).astype(np.float32)
    t = rng.normal(size=(4, 3)).astype(np.float32)
    loss = lambda: 0.5 * float(((net.forward(x) - t) ** 2).sum())
    out = net.forward(x, train=True)
    grads = net.backward(out - t)
    eps = 1e-2
    for p, g in zip(net.params(), grads):
        for idx in list(np.ndindex(*p.shape))[:6]:
            old = p[idx]
            p[idx] = old + eps
            lp = loss()
            p[idx] = old - eps
            lm = loss()
            p[idx] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - g[idx]) < 5e-2 * (1 + abs(num))


def test_adam_aprende_una_funcion_simple():
    net = MLP([2, 16, 1], seed=0)
    opt = Adam(net.params(), lr=1e-2)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (256, 2)).astype(np.float32)
    y = (x[:, :1] * 2 - x[:, 1:]).astype(np.float32)
    first = None
    for _ in range(300):
        out = net.forward(x, train=True)
        err = out - y
        first = first if first is not None else float((err ** 2).mean())
        opt.step(net.backward(err / len(x)))
    assert float((err ** 2).mean()) < first * 0.1


def test_guardar_y_cargar(tmp_path):
    net = MLP([3, 4, 2], seed=2)
    path = str(tmp_path / "m.npz")
    net.save(path)
    other = MLP.load(path)
    x = np.ones((1, 3), np.float32)
    assert np.allclose(net.forward(x), other.forward(x))
