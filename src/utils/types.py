from addict import Dict


class ForceKeyErrorDict(Dict):
    def __missing__(self, key):
        # raise KeyError(key)
        print("miss key: ", key)
