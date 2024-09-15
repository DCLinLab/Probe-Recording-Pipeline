from .rhd import read_data


class ItanSeriesLoader:
    def __init__(self, paths, no_floats=False):
        self.paths = paths
        self.no_floats = no_floats

    def __iter__(self):
        for path in self.paths:
            try:
                with open(path, 'r') as f:
                    yield read_data(path, no_floats=self.no_floats)
            except IOError as e:
                print(f"Error reading itan {path}: {e}")
