# A minimal, reusable registry.
class Registry(dict):
    def register(self, name=None):
        def _wrapper(fn):
            key = name or fn.__name__
            if key in self:
                raise KeyError(f"{key} already registered")
            self[key] = fn
            return fn
        return _wrapper
