import random
import time

import numpy as np

from typing import List, Set
from sklearn.linear_model import LinearRegression
from item import Item


# Returns the index of @item in @items using binary search.
# If @item does not exist in @items, returns the index of nearest item.
def search(items: List[Item], item: float) -> int:
  lo = 0
  hi = len(items)

  while lo < hi:
    idx = (lo + hi) >> 1

    if item < items[idx]._prophecy:
      lo = idx + 1
    elif items[idx]._prophecy < item:
      hi = idx
    else:
      return idx

  if lo == 0:
    return 0

  if lo == len(items):
    return lo - 1

  if abs(items[lo - 1]._prophecy - item) < abs(items[lo]._prophecy - item):
    return lo - 1

  return lo


# Finds least-filled non-empty pivot in @items and @acc
def least(items: List[List[Item]], acc: List[float]):
  _least = 0
  _min = acc[0]
  _max = acc[0]
  _diff = 0.0

  for rank in range(1, len(items)):
    if len(items[rank]) != 0 and acc[rank] <= _min:
      _least = rank
      _min = acc[rank]
      _diff = _max - _min
    elif _max < acc[rank]:
      _max = acc[rank]
      _diff = _max - _min

  # if pivot is empty, select next-least one
  if len(items[_least]) == 0:
    _least = 1
    _min = acc[1]
    for rank in range(2, len(items)):
      if len(items[rank]) != 0 and acc[rank] <= _min:
        _least = rank
        _min = acc[rank]
        _diff = _max - _min

  return _least, _diff


# Checks if @items are all empty.
def all_empty(items: List[List[Item]]) -> bool:
  for item in items:
    if len(item) != 0:
      return False

  return True


# TODO: use NumPy NDArray instead of Python list
class Scheduler(object):

  def __init__(self, n_task: int, batch_size: int):
    # global batch size
    self._batch_size = batch_size

    # number of tasks
    # this may be informed through either TF_CONFIG or strategy.num_replicas_in_sync.
    self._n_task = n_task

    # performance indicators of each task
    self._coefs = [1.0 for _ in range(self._n_task)]
    self._intercepts = [0.0 for _ in range(self._n_task)]

    # sum of length of scheduled data and elapsed time
    self._len_lists: List[List[List[int]]] = [[] for _ in range(self._n_task)]
    self._time_lists: List[List[List[float]]] = [[]
                                                 for _ in range(self._n_task)]

    self._item_lists: List[List[Item]] = [[] for _ in range(self._n_task)]
    self._scheduled: List[Set[Item]] = [set() for _ in range(self._n_task)]
    self._indices: List[List[int]] = [[] for _ in range(self._n_task)]

    # shortest items' indices for preventing ValueError when batch size = 0
    self._shortest_indices: List[int] = []

  def init(self, lens: List[int]):
    for index, length in enumerate(lens):
      # items are distributed in round-robin fashion initially
      # this may be redistributed in shuffling phase
      pivot_rank = index % self._n_task
      item = Item(index, length)
      item.update_prophecy(self._coefs[pivot_rank],
                           self._intercepts[pivot_rank])
      self._item_lists[pivot_rank].append(item)

    self.shuffle()
    self._shortest_indices = [
        item_list[-1]._index for item_list in self._item_lists
    ]

  # Stores sum of length of scheduled data and elapsed time.
  def feedback(self, rank, l, t):
    self._len_lists[rank].append([l])
    self._time_lists[rank].append([t])

  # Evaluates performance of @rank'th task based on feedbacks using linear regression.
  # This may be called after each mini-batch ends.
  def evaluate(self, rank):
    reg = LinearRegression().fit(self._len_lists[rank], self._time_lists[rank])
    self._coefs[rank] = reg.coef_[0][0]
    self._intercepts[rank] = reg.intercept_[0]

    # update prophecies and sort in descending order
    for item in self._item_lists[rank]:
      item.update_prophecy(self._coefs[rank], self._intercepts[rank])

    self._item_lists[rank].sort(key=lambda item: item._prophecy, reverse=True)

    if len(self._item_lists[rank]) != 0:
      self._shortest_indices[rank] = self._item_lists[rank][-1]._index

  # Schedules next mini-batch indices of each task using first-fit-decreasing.
  def schedule(self):
    indices: List[List[int]] = [[] for _ in range(self._n_task)]
    acc = [0.0 for _ in range(self._n_task)]
    progress = 0

    while progress < self._batch_size:
      # if all items are empty, ends scheduling
      # else if some items are empty, select non-empty node with least acc
      if all_empty(self._item_lists):
        # allocate at least one item to prevent ValueError
        for rank in range(self._n_task):
          if len(indices[rank]) == 0:
            indices[rank].append(self._shortest_indices[rank])

        self._indices = indices
        return

      pivot_rank = 0
      pivot_index = 0

      # if yet scheduled, select a random pivot
      if progress == 0:
        # select a non-empty node
        while pivot_rank < self._n_task and len(
            self._item_lists[pivot_rank]) == 0:
          pivot_rank += 1

        # select a random index
        random.seed(round(time.time()))
        pivot_index = random.randrange(0, len(self._item_lists[pivot_rank]))

      else:
        pivot_rank, diff = least(self._item_lists, acc)
        pivot_index = search(self._item_lists[pivot_rank], diff)

      indices[pivot_rank].append(self._item_lists[pivot_rank][pivot_index]._index)
      acc[pivot_rank] += self._item_lists[pivot_rank][pivot_index]._prophecy
      self._scheduled[pivot_rank].add(self._item_lists[pivot_rank][pivot_index])
      self._item_lists[pivot_rank] = self._item_lists[pivot_rank][:pivot_index] + self._item_lists[pivot_rank][pivot_index + 1:]

      progress += 1

    # allocate at least one item to prevent ValueError
    for rank in range(self._n_task):
      if len(indices[rank]) == 0:
        indices[rank].append(self._shortest_indices[rank])

    self._indices = indices

  # Redistributes items based on performance of each task using first-fit-decreasing.
  # This may be called before each epoch begins.
  def shuffle(self):
    sums: List[float] = []

    for item_list in self._item_lists:
      sums.append(sum([item._prophecy for item in item_list]))

    mean = np.mean(sums)

    # redistribute overflowed items
    overflows: List[Item] = []

    for rank in range(self._n_task):
      if mean < sums[rank]:
        extracted = self._extract(rank, sums[rank] - mean)
        sums[rank] -= sum([item._prophecy for item in extracted])
        overflows.extend(extracted)

    overflows.sort(key=lambda item: item._len, reverse=True)

    for item in overflows:
      pivot_rank = np.argmin(sums)
      item.update_prophecy(self._coefs[pivot_rank],
                           self._intercepts[pivot_rank])
      self._item_lists[pivot_rank].append(item)
      sums[pivot_rank] += item._prophecy

    for item_list in self._item_lists:
      item_list.sort(key=lambda item: item._prophecy, reverse=True)

  def _extract(self, rank, prophecy):
    extracted: List[Item] = []

    while 0 < prophecy:
      pivot_index = search(self._item_lists[rank], prophecy)
      prophecy -= self._item_lists[rank][pivot_index]._prophecy
      extracted.append(self._item_lists[rank][pivot_index])
      self._item_lists[rank] = self._item_lists[
          rank][:pivot_index] + self._item_lists[rank][pivot_index + 1:]

    return extracted

  # Resets item lists.
  def reset(self):
    for item_list, scheduled_set in zip(self._item_lists, self._scheduled):
      item_list.extend(list(scheduled_set))
    self._scheduled = [set() for _ in range(self._n_task)]
