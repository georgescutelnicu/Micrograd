# Create a neuron
class Neuron:
  
  def __init__(self, _in):
    self.w = [Value(np.random.uniform(-1, 1)) for _ in range(_in)]
    self.b = Value(np.random.uniform(-1, 1))
  
  def __call__(self, x):
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out
  
  def params(self):
    return self.w + [self.b]

# Create a layer of neurons
class Layer:
  
  def __init__(self, _in, _out):
    self.neurons = [Neuron(_in) for _ in range(_out)]
  
  def __call__(self, x):
    _out = [n(x) for n in self.neurons]
    return _out[0] if len(_out) == 1 else _out
  
  def params(self):
    return [p for neuron in self.neurons for p in neuron.params()]

# Create a Feedforward Neural Network
class FNN:
  
  def __init__(self, _in, __out):
    size = [_in] + __out
    self.layers = [Layer(size[i], size[i+1]) for i in range(len(__out))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def params(self):
    return [p for layer in self.layers for p in layer.params()]
