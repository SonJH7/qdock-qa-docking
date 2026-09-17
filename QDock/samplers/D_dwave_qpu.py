from .D_base import BaseSampler
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite

class DWaveQPUSamplerAdapter(BaseSampler):
    def __init__(self, solver=None, endpoint=None, token=None, **kwargs):
        super().__init__()
        # Avoid diskcache errors when solver cache DB is not writable/locked
        try:
            from dwave.cloud.client import base as _dwave_base
            _dwave_base.Client._DEFAULT_SOLVERS_CACHE_CONFIG = {"enabled": False}
        except Exception:
            pass
        self._base_sampler = DWaveSampler(solver=solver, endpoint=endpoint, token=token, **kwargs)
        self._sampler = EmbeddingComposite(self._base_sampler)
        self._solver_name = None
        try:
            self._solver_name = self._base_sampler.solver.name
        except Exception:
            self._solver_name = solver

    def sample_qubo(self, qubo, num_reads=None, **kwargs):
        embedding = kwargs.pop("embedding", None)
        if embedding is not None:
            kwargs.pop("embedding_parameters", None)
            sampler = FixedEmbeddingComposite(self._base_sampler, embedding)
        else:
            sampler = self._sampler
        if num_reads is None:
            num_reads = kwargs.pop("num_reads", None)
        if num_reads is not None:
            kwargs["num_reads"] = num_reads
        sampleset = sampler.sample_qubo(qubo, **kwargs)
        meta = {
            "backend": "dwave_qpu",
            "solver": self._solver_name,
            "num_reads": num_reads,
        }
        info = getattr(sampleset, "info", {}) or {}
        embedding = info.get("embedding_context", {}).get("embedding", {})
        if embedding:
            try:
                meta["physical_qubits"] = sum(len(v) for v in embedding.values())
                meta["max_chain_length"] = max(len(v) for v in embedding.values())
            except Exception:
                pass
        self._last_meta = meta
        return sampleset
