import json
import time
import threading

from flask import Flask, request
from scheduler import Scheduler

app = Flask(__name__)

scheduler = Scheduler(n_task=2, batch_size=4)

scheduled = False

# TODO: use bit mask instead of boolean list
evaluateds = [False for _ in range(scheduler._n_task)]
scheduleds = [False for _ in range(scheduler._n_task)]

# TODO: use RWLock
lock = threading.Lock()


def all_evaluated():
  for evaluated in evaluateds:
    if not evaluated:
      return False
  return True


def all_scheduled():
  for scheduled in scheduleds:
    if not scheduled:
      return False
  return True


@app.route("/init", methods=["POST"])
def init():
  # Initializes item lists using given length list.
  # This may be invoked only by the master.
  global scheduler

  lens = json.loads(request.get_data())["lens"]
  scheduler.init(lens)

  return ""


@app.route("/shuffle", methods=["GET"])
def shuffle():
  # Shuffles item lists.
  # This may be invoked only by the master.
  global scheduler
  global lock

  lock.acquire(blocking=True)
  scheduler.reset()
  scheduler.shuffle()
  lock.release()

  return ""


@app.route("/schedule", methods=["POST"])
def schedule():
  # Take feedbacks and update performance indicators.
  # Schedules next mini-batch indices.
  # This may be invoked by all workers.
  global scheduler
  global scheduled
  global scheduleds
  global evaluateds
  global lock

  _feed = json.loads(request.get_data())
  _rank = _feed["rank"]
  _len = _feed["len"]
  _time = _feed["time"]

  scheduler.feedback(_rank, _len, _time)
  scheduler.evaluate(_rank)

  lock.acquire(blocking=True)
  evaluateds[_rank] = True
  lock.release()

  if _rank == 0:
    # spins until performance evaluation is done
    # TODO: use condition variable to wake up early
    while not all_evaluated():
      time.sleep(0.0)
    lock.acquire(blocking=True)
    # FIXME: should schedule() be locked?
    scheduler.schedule()
    scheduled = True
    lock.release()

  # spins until scheduling is done
  # TODO: use condition variable to wake up early
  while not scheduled:
    time.sleep(0.0)

  lock.acquire(blocking=True)
  scheduleds[_rank] = True
  lock.release()

  if _rank == 0:
    # TODO: use condition variable to wake up early
    while not all_scheduled():
      time.sleep(0.0)

    lock.acquire(blocking=True)
    for rank in range(scheduler._n_task):
      evaluateds[rank] = False
      scheduleds[rank] = False
    scheduled = False
    lock.release()

  return json.dumps({"indices": scheduler._indices[_rank]})


app.run(host="0.0.0.0", port=5003, debug=False)
