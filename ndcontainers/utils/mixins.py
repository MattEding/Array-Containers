import inspect
import itertools


class NDArrayReprMixin:
    def _name_params(self):
        sig = inspect.signature(type(self))
        params = tuple(sig.parameters)
        name = type(self).__name__
        return name, params

    def _repr_(self, *param_values):
        name, params = self._name_params()

        head = params[0]
        width = len(name) + len(head) + 1
        tail = (p.rjust(width) for p in params[1:])
        pair_seq = itertools.chain.from_iterable(zip(tail, param_values[1:]))

        text = "{}(" +  ",\n".join("{}={}" for p in params) + ")"
        fmts = itertools.chain((name, head, param_values[0]), pair_seq)
        return text.format(*fmts)

    def _prefix_(self, param_idx):
        name, params = self._name_params()
        return f"{name}({params[param_idx]}="
