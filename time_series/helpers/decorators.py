from functools import wraps


class OnEnter:

    def __init__(self, g: callable):
        self.g = g

    def __call__(self, f: callable):
        @wraps(f)
        def closure(*args, **kwargs):
            self.g(f.__name__, args, kwargs)
            return f(*args, **kwargs)
        return closure


class OnExit:

    def __init__(self, g: callable):
        self.g = g

    def __call__(self, f: callable):
        @wraps(f)
        def closure(*args, **kwargs):
            result = f(*args, **kwargs)
            self.g(result)
            return result
        return closure


class OnException:

    def __init__(self, g: callable):
        self.g = g

    def __call__(self, f: callable):
        @wraps(f)
        def closure(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                self.g(e)
                raise e
        return closure