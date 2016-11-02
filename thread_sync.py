from threading import Event


class ThreadSync:

    e = Event()

    @classmethod
    def wait(cls):
        cls.e.wait()

    @classmethod
    def set(cls):
        cls.e.set()

    @classmethod
    def clear(cls):
        cls.e.clear()