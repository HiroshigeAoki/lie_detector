from src.model.AbstractModel import AbstractModel


class AbstractHierModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()     

    def build_wordattennet(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def build_sentattennet(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")