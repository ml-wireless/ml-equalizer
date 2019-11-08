class TrainInfo(object):
    def __init__(self, dataset, inputs, output, loss_func):
        self.dataset = dataset
        self.inputs = inputs
        self.outputs = output
        self.loss_func = loss_func
