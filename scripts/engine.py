class Value:
  
  def __init__(self, data, children=(), operation='', label=''):
    self.data = data
    self.children = children
    self.operation = operation
    self.label = label
    self.grad = 0
    self.derivative = lambda: None

  # Create a printable representation
  def __repr__(self):
    return f"Value({self.data})"
  
  # Self + Other
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    
    def derivative():
      self.grad += 1 * out.grad
      other.grad += 1 * out.grad
    out.derivative = derivative
    
    return out

  # Other + Self
  def __radd__(self, other): # other + self
    return self + other

  # Self * Other
  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def derivative():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out.derivative = derivative
      
    return out

  # Other * Self
  def __rmul__(self, other): # other * self
    return self * other
  
  # Self ** Other
  def __pow__(self, other):
    out = Value(self.data**other, (self, ), f'**{other}')

    def derivative():
      self.grad += other * (self.data ** (other - 1)) * out.grad
    out.derivative = derivative

    return out
  
  # Self / Other
  def __truediv__(self, other):
    return self / other

  # Negate self: self --> -self
  def __neg__(self):
    return self * -1

  # Self - Other
  def __sub__(self, other):
    return self + (-other)

  # Exponent
  def exp(self):
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')
    
    def derivative():
      self.grad += out.data * out.grad
    out.derivative = derivative
    
    return out

  # Activation fn --> tanh
  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')
    
    def derivative():
      self.grad += (1 - t**2) * out.grad
    out.derivative = derivative
    
    return out
  
  # Backward fn
  def backward(self):

    # Topological order
    topo = [] 
    visited = set() 
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v.children:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    # Initialize gradient
    self.grad = 1.0

    # Perform backpropagation
    for node in reversed(topo):
      node.derivative()
