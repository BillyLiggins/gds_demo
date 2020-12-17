from functools import wraps
from typing import Callable


def singleton(func: Callable) -> Callable:
    """Decorator that ensures only a single instance is available.
    While it can be used to decorate a class, it is not recommended,
    as it returns a function - you will no longer be able to access
    class attributes/methods, other than through the instance.
    Combine with functools.partial for maximum effect.
    Usage::
        @singleton
        def get_shared_service():
            return SomeServiceImpl(settings.SOME_SERVICE)
    :param func: A function or constructor that takes no arguments
    :return: The function, wrapped to return the same instance each time
    """
    unset = object()
    instance = unset

    @wraps(func)
    def wrapped():
        nonlocal instance
        if instance is unset:
            instance = func()
        return instance

    return wrapped
