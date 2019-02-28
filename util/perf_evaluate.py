import tensorflow as tf
import time, datetime, math

NUM_STEPS_BURN_IN = 1
NUM_BATCHES = 100

def time_run_mobilenet(framework, model=None, input=None, session=None, target=None, feed_dict=None, info_string=None):
  assert (framework in ['tensorflow', 'pytorch', 'keras'])
  total_duration = 0.0
  total_duration_squared = 0.0

  for i in range(NUM_BATCHES + NUM_STEPS_BURN_IN):
    start_time = time.time()
    if 'tensorflow' == framework:
      _ = session.run(target, feed_dict=feed_dict)
    elif 'pytorch' == framework:
      _ = model(input)
    elif 'keras' == framework:
      _ = model.predict(input)
    duration = time.time() - start_time
    if i == 0:
      print('%s across first run, %.3f' % (info_string, duration * 1000))
    if i >= NUM_STEPS_BURN_IN:
      # if not i % 10:
        # print('%s: step %d, duration = %.3f' % (datetime.now(), i - NUM_STEPS_BURN_IN, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / NUM_BATCHES
  vr = total_duration_squared / NUM_BATCHES - mn * mn
  sd = math.sqrt(vr)
  print('%s across %d steps, %.3f +/- %.3f ms / batch' % (info_string, NUM_BATCHES, mn*1000, sd*1000))



