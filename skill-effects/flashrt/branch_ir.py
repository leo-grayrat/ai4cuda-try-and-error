"""Research-controller IR for validating the supplied two-branch reference.

This helper is not staged into a subject workspace; subjects build their own IR.
"""


def build_graph(repository, config):
    import torch
    import torch.nn.functional as F
    from branch_features.model import build_layers
    from flashrt.ir import IRGraph, IREdge, IROperator, OpType, TensorPort, StateObject

    class Convolution(IROperator):
        def __init__(self, name, layer, activate):
            self.layer, self.activate, self.history = layer, activate, 'history_' + name
            super().__init__(name=name, op_type=OpType.EXPOSED, inputs=[TensorPort('features')],
                             outputs=[TensorPort('features')], state_reads=[self.history], state_writes=[self.history])

        def execute(self, inputs, context, state):
            features, previous = inputs['features'], state.get(self.history)
            combined = features if previous is None else torch.cat([previous, features], dim=-1)
            output = self.layer(combined)
            if previous is not None:
                output = output[:, :, previous.shape[-1]:]
            state.set(self.history, combined[:, :, -self.layer._time_causal_padding[0]:].clone())
            return {'features': F.silu(output) if self.activate else output}

    class Gate(IROperator):
        def execute(self, inputs, context, state):
            return {'gate': torch.sigmoid(inputs['features'].mean(dim=1, keepdim=True))}

    class Merge(IROperator):
        def execute(self, inputs, context, state):
            gate = inputs['gate']
            return {'features': gate * inputs['a'] + (1 - gate) * inputs['b'] + 0.1 * inputs['features']}

    layers = build_layers(repository, config, 'cpu')
    graph = IRGraph(name='branch_features_reference', external_inputs=[TensorPort('features')],
                    external_outputs=[TensorPort('features')])
    for name, layer in layers.items():
        graph.add_operator(Convolution(name, layer, name.endswith('0')))
        graph.add_state(StateObject(name='history_' + name))
    graph.add_operator(Gate(name='gate', op_type=OpType.EXPOSED, inputs=[TensorPort('features')], outputs=[TensorPort('gate')]))
    graph.add_operator(Merge(name='merge', op_type=OpType.EXPOSED,
                             inputs=[TensorPort(name) for name in ['features', 'a', 'b', 'gate']], outputs=[TensorPort('features')]))
    graph.chunk_ops = [*layers.keys(), 'gate', 'merge']
    for source, source_port, target, target_port in [
        ('a0', 'features', 'a1', 'features'), ('b0', 'features', 'b1', 'features'),
        ('a1', 'features', 'merge', 'a'), ('b1', 'features', 'merge', 'b'), ('gate', 'gate', 'merge', 'gate')]:
        graph.add_edge(IREdge(source, source_port, target, target_port))
    return graph
