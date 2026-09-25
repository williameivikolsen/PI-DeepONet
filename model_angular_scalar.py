from model import PI_DeepONet_Angular


class PI_DeepONet_AngularScalar(PI_DeepONet_Angular):
    """
    Same architecture as PI_DeepONet_Angular, but trained with scalar flux data loss
    """
    def data_net(self, params, Q, x):
        return self.phi0_net(params, Q, x)
