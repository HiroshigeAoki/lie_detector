from src.model.AbstractModel import AbstractModel


class AbstractHierAttnNet(AbstractModel):
    def __init__(self, optim: dict):
        super().__init__(optim=optim)
        self.save_hyperparameters()

    
    def build_wordattennet(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")


    def build_sentattennet(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")