# mainthread.py
import queue

main_thread_queue = queue.Queue()

def queue_main_thread(func):
    main_thread_queue.put(func)
