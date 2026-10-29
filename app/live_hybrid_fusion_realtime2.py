"""Capture class of the DentWise acquisition software (Elgato HD60 X capture card).

This module contains only `HD60XOptimizedCapture`, copied unchanged from the authors' file of the same name, so that
`gui_pyqt6.py` and `live_unified.py` import it as before. The rest of that file (an earlier landmark-model and
training code) is not part of this repository.
"""
import queue
import threading

import cv2


class HD60XOptimizedCapture:
    """Optimized for Elgato HD60 X capture card"""
    def __init__(self, device_id=0, crop_x=800, crop_y=220, crop_width=610, crop_height=640):
        self.device_id = device_id
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.capture_thread = None
        self.last_frame = None
        
    def start_capture(self):
        print(f"Initializing HD60 X (device {self.device_id})...")
        
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            print("MSMF failed, trying DSHOW...")
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open HD60 X on device {self.device_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1924)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 980)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        yuv2_codec = cv2.VideoWriter_fourcc(*'YUY2')
        self.cap.set(cv2.CAP_PROP_FOURCC, yuv2_codec)
        
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            print("YUY2 failed, trying MJPG...")
            mjpg_codec = cv2.VideoWriter_fourcc(*'MJPG')
            self.cap.set(cv2.CAP_PROP_FOURCC, mjpg_codec)
            ret, test_frame = self.cap.read()
        
        if ret and test_frame is not None:
            self.last_frame = test_frame
            print("HD60 X initialized")
        else:
            raise ValueError("Cannot read frames from HD60 X")
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Capture: {width}x{height} @ {fps} FPS")
        
        if self.crop_x + self.crop_width > width:
            self.crop_width = width - self.crop_x
        if self.crop_y + self.crop_height > height:
            self.crop_height = height - self.crop_y
        
        print(f"Crop: {self.crop_width}x{self.crop_height}")
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._aggressive_capture_loop, daemon=True)
        self.capture_thread.start()
    
    def _aggressive_capture_loop(self):
        consecutive_failures = 0
        
        while self.is_running and consecutive_failures < 30:
            ret = self.cap.grab()
            if not ret:
                consecutive_failures += 1
                continue
            
            ret, frame = self.cap.retrieve()
            if not ret or frame is None:
                consecutive_failures += 1
                continue
            
            consecutive_failures = 0
            
            cropped = frame[self.crop_y:self.crop_y+self.crop_height, 
                          self.crop_x:self.crop_x+self.crop_width]
            
            self.last_frame = cropped
            
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    break
            
            try:
                self.frame_queue.put(cropped, block=False)
            except:
                pass
        
        if consecutive_failures >= 30:
            print("HD60 X connection lost")
            self.is_running = False
    
    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=0.05)
        except queue.Empty:
            return self.last_frame
    
    def stop_capture(self):
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
