class SGD:

    def __init__(
        self,
        parameters,
        learning_rate=0.05
    ):

        self.parameters = parameters

        self.learning_rate = learning_rate

    def step(self, gradients):

        for parameter, gradient in zip(
            self.parameters,
            gradients
        ):

            parameter -= (
                self.learning_rate
                * gradient
            )