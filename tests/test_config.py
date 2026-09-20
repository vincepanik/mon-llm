from utils import get_device, load_config


def test_debug_config_loads():
    cfg = load_config("configs/debug_mac.py")
    assert cfg.run_name == "debug"
    assert 0 < cfg.throttle <= 1


def test_run_config_loads():
    cfg = load_config("configs/run_150m.py")
    assert cfg.n_embd % cfg.n_head == 0


def test_device():
    assert get_device().type in ("cuda", "mps", "cpu")
