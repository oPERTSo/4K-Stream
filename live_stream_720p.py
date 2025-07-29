import cv2
import subprocess
import numpy as np
import time
import threading
import queue

# HLS stream URL
url = "http://202.151.178.122/boxfilm/index.m3u8"

# FFmpeg command to read and scale video to 720p (CPU)
ffmpeg_cmd = [
    "ffmpeg",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-threads", "4",
    "-i", url,
    "-vf", "scale=1280:720",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-an", "-sn", "-dn",
    "-loglevel", "error",
    "-"
]

# Start FFmpeg process
proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=10**8)

width, height = 1280, 720
frame_size = width * height * 3

# Frame buffer for smooth playback
frame_buffer = queue.Queue(maxsize=60)  # buffer up to 60 frames (ลด latency เพิ่มความสดใหม่)



def read_frames():
    while True:
        raw_frame = proc.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            time.sleep(0.0001)
            continue
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        # ถ้า buffer เต็มมาก ๆ ให้ drop เฟรมเก่า 3 ครั้ง เพื่อกัน buffer overflow
        if frame_buffer.full():
            for _ in range(3):
                if frame_buffer.full():
                    frame_buffer.get()
        frame_buffer.put(frame)


# Start thread to read frames into buffer
reader_thread = threading.Thread(target=read_frames, daemon=True)
reader_thread.start()






# จำกัดอัตราเฟรมที่ 25 fps เพื่อให้ภาพไม่ไวเกินไป
target_fps = 25
frame_interval = 1.0 / target_fps
last_frame_time = time.time()

cv2.namedWindow("Live Stream 720p (Sharpen)", cv2.WINDOW_NORMAL)

while True:
    # ถ้า buffer เหลือน้อยกว่า 3 เฟรม ให้รอเพื่อป้องกันภาพ freeze
    if frame_buffer.qsize() < 3:
        time.sleep(0.0001)
        continue
    # ถ้า buffer เกิน 50 เฟรม ให้ข้ามเฟรม (frame skip) เพื่อความสดใหม่
    if frame_buffer.qsize() > 50:
        for _ in range(2):
            if not frame_buffer.empty():
                frame_buffer.get()
    now = time.time()
    elapsed = now - last_frame_time
    if elapsed < frame_interval:
        time.sleep(frame_interval - elapsed)
    last_frame_time = time.time()
    if not frame_buffer.empty():
        frame = frame_buffer.get()
        # เพิ่มความคมชัดด้วย Sharpen filter (เปิดใช้งานอีกครั้ง)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(frame, -1, kernel)
        cv2.imshow("Live Stream 720p (Sharpen)", sharpened)
    else:
        time.sleep(0.0001)
        continue
    if cv2.waitKey(1) == 27:
        break

proc.terminate()
cv2.destroyAllWindows()
