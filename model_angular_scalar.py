from model import PI_DeepONet, PI_DeepONet_Angular


class PI_DeepONet_AngularScalar(PI_DeepONet_Angular):
    """
    Same architecture as PI_DeepONet_Angular: trunk takes x only, emits
    A*p outputs reshaped to (A, p) and contracted with the branch to
    produce all A angular-flux channels from a single forward pass
    (angular_net / operator_net, both inherited unchanged).

    Unlike PI_DeepONet_Angular, this class is supervised on scalar flux
    phi_0 targets instead of the full angular vector psi: loss_data is
    restored to PI_DeepONet's phi_0-via-GL-quadrature form (bypassing
    PI_DeepONet_Angular's vector-MSE-on-psi override). This isolates the
    architecture choice (vector-output trunk vs. joint (x,mu) trunk) from
    the label choice (phi_0 vs psi) when compared against PI_DeepONet and
    PI_DeepONet_Angular.
    """
    def loss_data(self, params, batch):
        return PI_DeepONet.loss_data(self, params, batch)
