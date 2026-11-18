# ruff: noqa
"""Minimal torch stub so agents.py imports without a 2 GB wheel."""


class _Cuda:
    @staticmethod
    def is_available():
        return False


class _MPS:
    @staticmethod
    def is_available():
        return False


class _Backends:
    mps = _MPS()


class _Version:
    hip = None
    cuda = None


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


version = _Version()
cuda = _Cuda()
backends = _Backends()


def no_grad():
    return _NoGrad()
