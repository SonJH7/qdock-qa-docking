class BaseSampler:
    def __init__(self):
        self._last_meta = {}

    def sample_qubo(self, qubo, num_reads=None, **kwargs):
        raise NotImplementedError

    def get_meta(self):
        return dict(self._last_meta) if self._last_meta else {}
