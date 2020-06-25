# xparl can be used to accelerate training process.
# In a multi-core computer, one can also use multiprocessing to do parallel training.
# xparl can do more, it can help you leverage both the local and the remote HPC's power.

import threading
import parl
import time

st = time.time()

@parl.remote_class                 # 1. decorate
class A(object):
    def run(self):
        ans = 0
        for i in range(100000000):
            ans += i
threads = []

parl.connect("localhost:6006")    # 2. connect to xparl
for _ in range(5):
    a = A()
    th = threading.Thread(target=a.run)
    th.start()
    threads.append(th)
for th in threads:
    th.join()

et = time.time()

print(f'{et-st}')