class MultiplyGate(object):
    def forward(x,y):
        z = x*y
        return z

    def backward(dz):
        dx = self.y * dz  # [dz/dx * dL/dz]
        dy = self.x * dz  # [dz/dx * dL/dz]
        return [dx, dy]