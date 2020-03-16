class ComputationalGraph(object):

    def forwrad(inputs):
        # 1. [pass inputs to input gates...]
        # 2. forward the computational graph:
        for gate in self.graph.nodes_topolohically_sorted():
            gate.forward()
        return loss # the final gate in the graph outputs the loss

    # 미분인 것 같음
    def backward(self):
        for gate in reversed(self.graph.nodes_topolohically_sorted()):
            gate.backward()  # little piece of backprop (chain rule applied)
        return inputs_gradients
