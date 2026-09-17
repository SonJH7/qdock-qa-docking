import json
import os

from .D_base import BaseSampler

class FallbackSampler(BaseSampler):
    def __init__(self, samplers):
        super().__init__()
        self.samplers = samplers

    def sample_qubo(self, qubo, num_reads=None, **kwargs):
        last_err = None
        for sampler in self.samplers:
            try:
                result = sampler.sample_qubo(qubo, num_reads=num_reads, **kwargs)
                meta = sampler.get_meta() if hasattr(sampler, "get_meta") else {}
                if meta:
                    meta = dict(meta)
                    meta.setdefault("fallback_chain", [type(s).__name__ for s in self.samplers])
                self._last_meta = meta
                return result
            except Exception as exc:
                last_err = exc
        if last_err:
            raise last_err
        raise RuntimeError("No sampler available")

class SamplerFactory:
    @staticmethod
    def from_config(cfg):
        if isinstance(cfg, str):
            cfg = load_config(cfg)
        backend = cfg.get("backend", "neal")
        params = cfg.get("backend_params", {}) or {}
        fallback = cfg.get("fallback")
        sampler = build_sampler(backend, params)
        if fallback:
            samplers = [sampler]
            for item in fallback:
                if isinstance(item, str):
                    fb_backend, fb_params = item, {}
                elif isinstance(item, dict):
                    fb_backend = item.get("backend")
                    fb_params = item.get("backend_params", {}) or {}
                else:
                    continue
                samplers.append(build_sampler(fb_backend, fb_params))
            return FallbackSampler(samplers)
        return sampler

def build_sampler(backend, params=None):
    params = params or {}
    if backend == "neal":
        from .D_neal import NealSamplerAdapter
        return NealSamplerAdapter(**params)
    if backend == "dwave_qpu":
        from .D_dwave_qpu import DWaveQPUSamplerAdapter
        return DWaveQPUSamplerAdapter(**params)
    if backend == "dwave_hybrid":
        from .D_dwave_hybrid import DWaveHybridSamplerAdapter
        return DWaveHybridSamplerAdapter(**params)
    raise ValueError("Unknown backend: %s" % (backend,))

def load_config(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as f:
            return json.load(f)
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML not installed; use .json config or install pyyaml") from exc
    with open(path) as f:
        return yaml.safe_load(f)
