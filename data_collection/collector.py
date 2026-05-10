import time
import csv
import os
import threading
from pynput import mouse

class MouseCollector:
    def __init__(self, user_id, save_dir="collected_data", flush_interval=300):
        self.user_id = user_id
        self.save_dir = os.path.join(save_dir, user_id)
        self.flush_interval = flush_interval
        self.events = []
        self.lock = threading.Lock()
        self.session_start = time.time()
        self.running = False
        os.makedirs(self.save_dir, exist_ok=True)

    def elapsed(self):
        return time.time() - self.session_start

    def flush(self):
        with self.lock:
            if not self.events:
                return None
            events_to_save = self.events.copy()
            self.events = []
        timestamp = int(time.time())
        path = os.path.join(self.save_dir, f"session_{timestamp}.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(events_to_save)
        print(f"[Collector] Flushed {len(events_to_save)} events to {path}")
        return path
    
    def flush_loop(self, trainer=None):
        while self.running:
            time.sleep(self.flush_interval)
            path = self.flush()
            if path is not None and trainer is not None:
                trainer.update(path)
    
    def on_move(self, x, y):
        t = self.elapsed()
        with self.lock:
            self.events.append([t, t, "NoButton", "Move", x, y])

    def on_click(self, x, y, button, pressed):
        t = self.elapsed()
        btn = 'Left' if button == mouse.Button.left else 'Right'
        state = 'Pressed' if pressed else 'Released'
        with self.lock:
            self.events.append([t, t, btn, state, x, y])
    
    def on_scroll(self, x, y, dx, dy):
        t = self.elapsed()
        direction = 'Down' if dy < 0 else 'Up'
        with self.lock:
            self.events.append([t, t, 'NoButton', f'Scroll {direction}', 0, 0])
    
    def start(self, trainer=None):
        self.running = True
        flush_thread = threading.Thread(
            target=self.flush_loop,
            args=(trainer,),
            daemon=True
        )
        flush_thread.start()
        print(f"[Collector] Listening for user {self.user_id}... (Ctrl+C to stop)")
        with mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click
        ) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                self.running = False
                path = self.flush()
                if path is not None and trainer is not None:
                    trainer.update(path)
                print("[Collector] Stopped.")