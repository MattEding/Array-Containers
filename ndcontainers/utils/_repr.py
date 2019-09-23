import inspect
import textwrap


class ReprMixin:
    def __repr__(self):
        sig = inspect.signature(type(self))
        
