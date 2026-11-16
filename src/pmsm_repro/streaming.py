"""Numerical filter state copied unchanged from the recorded scalar measurement."""
import numpy as np
SPANS=np.asarray([16,64,256,1024,4096],dtype=np.float64)

class FeatureState:
    """Only required causal33 features; no unused 23-feature auxiliary builder."""
    def __init__(self, kind, arm, dt_ref=None):
        self.kind, self.arm, self.dt_ref = kind, arm, dt_ref
        self.state = None
        self.previous_time = None
        self.decay = 1. - 2. / (SPANS + 1.)
        self.log_decay = np.log1p(-2. / (SPANS + 1.))
        if kind == 'rwth' and (arm != 'original' or dt_ref is None or dt_ref <= 0):
            raise ValueError('RWTH original time-aware feature definition required')
        if kind == 'legacy' and arm not in ('original', 'own_boundary'):
            raise ValueError('Frozen original/own_boundary timing panels')

    def snapshot(self):
        return (None if self.state is None else self.state.copy(), self.previous_time)

    def restore(self, snapshot):
        self.state = None if snapshot[0] is None else snapshot[0].copy()
        self.previous_time = snapshot[1]

    def step(self, op, timestamp=None):
        if self.kind == 'legacy':
            current = np.hypot(op[0], op[1])
            speed = np.abs(op[4])
            voltage = np.hypot(op[2], op[3])
            instant = np.asarray([current, current ** 2, voltage, speed, speed ** 2,
                                  current * speed, op[5], op[6]], dtype=np.float64)
            drive = np.asarray([op[6], op[5], instant[1], instant[4], instant[5]])
            if self.state is None:
                self.state = np.zeros((5, 5), dtype=np.float64)  # span-major
                self.state[:, :2] = ([op[6], op[5]] if self.arm == 'own_boundary'
                                     else [op[6], op[6]])
            else:
                # Same multiplication/addition order as the frozen scipy lfilter.
                self.state = ((1. - self.decay)[:, None] * drive[None, :]
                              + self.decay[:, None] * self.state)
            return (self.state[:, :2].reshape(-1), self.state[:, 2:].reshape(-1), instant)
        current, rpm, torque, coolant = op
        omega = 2. * np.pi * rpm / 60.
        instant = np.asarray([current, omega, torque, coolant, current ** 2,
                              np.abs(omega), omega ** 2, np.abs(torque * omega)])
        drive = np.asarray([coolant, instant[4], instant[5], instant[6], instant[7]])
        if self.state is None:
            self.state = np.zeros((5, 5), dtype=np.float64)  # drive-major
            self.state[0] = coolant
        else:
            if timestamp <= self.previous_time:
                raise ValueError('Strictly increasing original timestamps required')
            alpha = -np.expm1(self.log_decay * ((timestamp - self.previous_time) / self.dt_ref))
            self.state = (1. - alpha[None, :]) * self.state + alpha[None, :] * drive[:, None]
        self.previous_time = timestamp
        return np.concatenate((instant, self.state.reshape(-1)))

def legacy_x(packet, scale):
    b, heat, instant = packet
    return np.concatenate((b, heat / np.asarray(scale['qscale']), instant))
