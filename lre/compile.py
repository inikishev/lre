"""copied from torchzero and this is for Newton-Schulz which is actually made faster by it"""
import time

import torch
import torch.utils.benchmark

class _OptionalCompiler:
    """this holds .enable attribute, set to True to enable compiling library wise"""
    def __init__(self):
        self.enable = False

    def enable_compilation(
        self,
        x,
        fullgraph: bool = False,
        dynamic: bool | None = None,
        backend="inductor",
        mode: str | None = "max-autotune-no-cudagraphs",
        options: dict[str, str | int | bool] | None = None,
        disable: bool = False,
    ):
        """compiles if self.compile is True otherwise returns uncompiled `x`"""
        return _MaybeCompiledFunc(x, self, fullgraph=fullgraph, dynamic=dynamic, backend=backend, mode=mode, options=options, disable=disable)

class _MaybeCompiledFunc:
    def __init__(self, func, compiler: _OptionalCompiler, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.compiled = False
        self.compiler = compiler

    def __call__(self, *args, **kwargs):
        if self.compiler.enable and not self.compiled:
            self.func = torch.compile(self.func, **self.kwargs)
            self.compiled = True
        return self.func(*args, **kwargs)


_optional_compiler = _OptionalCompiler()
"""this holds .enable attribute, set to True to enable compiling for a few functions that benefit from it."""

def enable_compilation(enable: bool=True):
    """`enable` is False by default. When True, certain functions will be compiled, which may not work on some systems like Windows, but it usually improves performance."""
    _optional_compiler.enable = enable

def allow_compile(fn): return _optional_compiler.enable_compilation(fn)
