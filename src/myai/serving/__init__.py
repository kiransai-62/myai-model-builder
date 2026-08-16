from .runtime import MyAIRuntime, InferenceRequest, InferenceResponse

def __getattr__(name: str):
    if name == "create_app":
        from .app import create_app
        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MyAIRuntime", "InferenceRequest", "InferenceResponse", "create_app"]
