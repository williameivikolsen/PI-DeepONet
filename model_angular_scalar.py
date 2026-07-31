from model import PI_DeepONet, PI_DeepONet_Angular


class PI_DeepONet_AngularScalar(PI_DeepONet_Angular):
    """
    Same architecture as PI_DeepONet_Angular, but trained with scalar flux data loss
    """
    def loss_data(self, params, batch):
        return PI_DeepONet.loss_data(self, params, batch)
