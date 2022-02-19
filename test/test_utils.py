import time


def wait_for_shutdown(distributed_model):
    while True:
        time.sleep(1)
        flg = True
        for q in distributed_model._update_queues:
            try:
                if distributed_model._update_queues[q].size() > 0:
                    flg = False
                    break
            except:
                # workers have finished
                return
        if flg:
            break
