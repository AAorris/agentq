"""Basic memory stream with random sampling."""


class MemoryStream(object):
    def __init__(self, size):
        self.memory = deque([])

    def add(self, item):
        self.memory.append(item)
        if len(self.memory) > size:
            memory.popleft()

    def sample(self, shape):
        return np.random.sample(memory, shape)
