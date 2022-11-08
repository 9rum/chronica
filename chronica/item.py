# Represents an item in Chronos Performance Prophet.
class Item(object):

  def __init__(self, index: int, length: int, prophecy: float = 1.0):
    self._index = index
    self._len = length
    self._prophecy = prophecy

  def update_prophecy(self, coef: float, intercept: float):
    self._prophecy = coef * self._len + intercept
