"""V100-specific Qwen4-Exp PLE host-offload initialization fix.

SGLang's model registry imports every module in this package. Importing this
module patches Qwen4ExpNGramEmbedding.__init__ so host-offloaded PLE tables are
created as metadata on the meta device and materialized directly in pinned host
memory by Qwen4ExpPinnedHostEmbedding.

The wrapper forwards every upstream keyword (including ``prefix``) so this
patch does not drift from Qwen4ExpNGramEmbedding.__init__.
"""

from sglang.srt.models import qwen4_exp as _base

_orig_ngram_embedding_init = _base.Qwen4ExpNGramEmbedding.__init__


def _v100_ngram_embedding_init(self, *args, **kwargs):
    config = args[0] if args else kwargs["config"]
    if not getattr(config, "ple_offload_embedding", False):
        return _orig_ngram_embedding_init(self, *args, **kwargs)

    orig_cls = _base.VocabParallelEmbedding

    def _meta_vpe(*a, **k):
        with _base.torch.device("meta"):
            return orig_cls(*a, **k)

    _base.VocabParallelEmbedding = _meta_vpe
    try:
        return _orig_ngram_embedding_init(self, *args, **kwargs)
    finally:
        _base.VocabParallelEmbedding = orig_cls


_base.Qwen4ExpNGramEmbedding.__init__ = _v100_ngram_embedding_init
