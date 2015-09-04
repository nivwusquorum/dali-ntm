import dali.core as D
import numpy as np

from dali.utils import Capture

def row_by_scalar(a,b):
    return D.MatOps.eltmul(a, b, broadcast=True, axis=0)

def col_by_scalar(a,b):
    return D.MatOps.eltmul(a, b, broadcast=True, axis=1)

class NTMAddressing(object):
    def __init__(self, input_sizes,
                       memory_locations,
                       memory_size,
                       shift_type='local'):
        self.memory_locations = memory_locations
        self.shift_type       = shift_type

        self.content_key          = D.StackedInputLayer(input_sizes, memory_size)
        self.content_key_strength = D.StackedInputLayer(input_sizes, 1)
        self.interpolation        = D.StackedInputLayer(input_sizes, 1)
        self.focus                = D.StackedInputLayer(input_sizes, 1)


        if shift_type == 'local':
            self.shift = D.StackedInputLayer(input_sizes, 3)
        else:
            raise Exception("Unknown shift mask type: %s" % (shift_type,))

        self.initial_locations = D.random.uniform(-0.1, 0.1, (1,self.memory_locations))

    def name_parameters(self, prefix):
        self.content_key.name_parameters(prefix + "_content_key")
        self.content_key_strength.name_parameters(prefix + "_content_key_strength")
        self.interpolation.name_parameters(prefix + "_interpolation")
        self.shift.name_parameters(prefix + "_shift")
        self.focus.name_parameters(prefix + "_focus")
        self.initial_locations.name = prefix + "_initial_locations"

    def content_addressing_activation(self, key, key_strength, memory):
        # cosine distance essentially
        key_broadcasted   = D.MatOps.broadcast(key, axis=0, num_replicas=memory.shape[0])
        unnormalized_dot  = (key * memory).sum(axis=1)
        key_norm          = key_broadcasted.L2_norm(axis=1)
        memory_norm       = memory.L2_norm(axis=1)
        cosine_similarity = (unnormalized_dot / (key_norm * memory_norm + 1e-6)).T()
        presoftmax        = row_by_scalar(cosine_similarity, key_strength)

        return D.MatOps.softmax(presoftmax)

    def shift_activation(self, inputs, weights):
        shift                = D.MatOps.softmax(self.shift.activate(inputs)).T()

        if self.shift_type == 'local':
            full_shift = D.MatOps.hstack([
                shift[0],
                shift[1],
                D.Mat.zeros((1, self.memory_locations - 3)),
                shift[2]
            ])
        else:
            assert False

        return D.MatOps.circular_convolution(weights, full_shift)

    def address(self, inputs, memory, state):
        """Outputs memory location weights.

        Inputs:
        inputs -- set of vectors controlling the mechanism (e.g. LSTM output)
        state  -- weights from previous timestep.
        """
        # todo - should memory contents be tanhed?
        key                  = self.content_key.activate(inputs)
        key_strength         = self.content_key_strength.activate(inputs).softplus()
        # todo - make multiplication of similarity (vector) * key strength_scalrar is correctly broadcasted
        content_weights      = self.content_addressing_activation(key, key_strength, memory)
        interpolation_gate   = self.interpolation.activate(inputs).sigmoid()
        interpolated_weights = (row_by_scalar(content_weights, interpolation_gate) +
                                row_by_scalar(state,1.0 - interpolation_gate))
        shifted_weighs       = self.shift_activation(inputs, interpolated_weights)
        focus                = self.focus.activate(inputs).softplus() + 1.
        focused_weights      = shifted_weighs ** focus
        # todo - make sure it's correctly broadcasted
        sum_focused          = focused_weights.sum(axis=1)
        focused_weights      = focused_weights / (sum_focused + 1e-6)

        return focused_weights

    def initial_states(self):
        return D.MatOps.softmax(self.initial_locations)

    def parameters(self):
        res = []
        res.extend(self.content_key.parameters())
        res.extend(self.content_key_strength.parameters())
        res.extend(self.interpolation.parameters())
        res.extend(self.shift.parameters())
        res.extend(self.focus.parameters())
        res.append(self.initial_locations)
        return res

class NTMReadHead(object):
    def __init__(self, input_sizes, memory_locations, memory_size):
        self.addressing = NTMAddressing(input_sizes, memory_locations, memory_size)

    def name_parameters(self, prefix):
        self.addressing.name_parameters(prefix + "_adressing")

    def read(self, inputs, memory, state):
        weights = self.addressing.address(inputs, memory, state)
        Capture.add("read_head_weights", weights)
        # todo - make sure it is correctly broadcasted
        pulled_from_memory = row_by_scalar(memory, weights.T()).sum(axis=0)
        Capture.add("read_head_content", pulled_from_memory)
        return pulled_from_memory, weights

    def initial_states(self):
        return self.addressing.initial_states()

    def parameters(self):
        return self.addressing.parameters()


class NTMWriteHead(object):
    def __init__(self, input_sizes, memory_locations, memory_size):
        self.addressing = NTMAddressing(input_sizes, memory_locations, memory_size)
        self.content    = D.StackedInputLayer(input_sizes, memory_size)
        self.erase      = D.StackedInputLayer(input_sizes, memory_size)

    def name_parameters(self, prefix):
        self.addressing.name_parameters(prefix + "_adressing")
        self.content.name_parameters(prefix + "_content")
        self.erase.name_parameters(prefix + "_erase")


    def write(self, inputs, memory, state):
        weights = self.addressing.address(inputs, memory, state)
        Capture.add("write_head_weights", weights)

        # todo - make sure it is correctly broadcasted
        new_content = self.content.activate(inputs)
        Capture.add("write_head_content", new_content)

        erase       = self.erase.activate(inputs).sigmoid()

        memory = memory * (1.0 - weights.T().dot(erase))
        # todo - make sure it is an outer product
        memory = memory + weights.T().dot(new_content)
        return memory, weights

    def initial_states(self):
        return self.addressing.initial_states()

    def parameters(self):
        return self.addressing.parameters()



__all__ = ['NTMAddressing', 'NTMReadHead', 'NTMWriteHead']
