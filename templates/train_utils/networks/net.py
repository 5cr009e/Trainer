class Network:
    def __init__(self, network):
        self.network = network

    def preprocess(self):
        pass

    def train(self, data_dict, optimizer):
        self.network.train()
        pass

    def eval(self, data_dict):
        self.network.eval()
        pass

    def get_loss(self, label, output):
        pass

    def save(self):
        pass
