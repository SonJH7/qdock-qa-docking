from .D_base import BaseSampler
import neal

class NealSamplerAdapter(BaseSampler):
    def __init__(self, seed=42):
        super().__init__()
        self.seed = seed
        self._sampler = neal.SimulatedAnnealingSampler()

    def sample_qubo(self, qubo, num_reads=None, **kwargs):
        if num_reads is None:
            num_reads = kwargs.pop("num_reads", None)
        if num_reads is not None:
            kwargs["num_reads"] = num_reads
        kwargs.setdefault("seed", self.seed)
        sampleset = self._sampler.sample_qubo(qubo, **kwargs)
        self._last_meta = {
            "backend": "neal",
            "num_reads": num_reads,
            "seed": kwargs.get("seed"),
        }
        return sampleset
