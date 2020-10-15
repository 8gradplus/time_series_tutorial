from helpers.decorators import OnEnter, OnExit, OnException


def on_enter(logger):
    msg = "Execute function '{name}' with arguments: {args} and key-word argument: {kwargs}"
    return OnEnter(lambda name, args, kwargs: logger.info(msg.format(name=name, args=args, kwargs=kwargs)))


def on_exit(logger):
    return OnExit(lambda r: logger.info("End computation of function with result {result} ".format(result=str(r))))


def on_exception(logger):
    return OnException(lambda e: logger.error("An exception has occured: {exception} ".format(exception=str(e))))



