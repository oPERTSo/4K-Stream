import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from tqdm import tqdm
import time
import torch

# ตั้งค่า URL ของ live stream
stream_url = 'http://10.4.3.222/t2-ch8/index.m3u8'

# เปิด stream
cap = cv2.VideoCapture(stream_url)


# อ่านค่า FPS จาก stream
stream_fps = cap.get(cv2.CAP_PROP_FPS)
if stream_fps is None or stream_fps <= 0 or stream_fps > 60:
    stream_fps = 25  # กำหนดค่าเริ่มต้นเป็น 25 fps (มาตรฐาน HLS/TV)

wait_ms = max(1, int(1000 / stream_fps))

# เตรียม VideoWriter สำหรับบันทึกวิดีโอที่ผ่านการ upscale
output_path = 'output_upscaled.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, stream_fps, (1280, 720))

# สร้าง RRDBNet สำหรับ x4
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

# โหลด RealESRGANer พร้อม model

# แจ้งเตือนถ้าใช้งานบน CPU

# ตรวจสอบว่ามี GPU (CUDA) หรือไม่
if torch.cuda.is_available():
    device = 'cuda'
    print('พบ GPU (CUDA) กำลังใช้งาน RealESRGAN บน GPU เพื่อความเร็วสูงสุด')
    print(torch.cuda.get_device_name(0))
else:
    device = 'cpu'
    print('คำเตือน: ไม่พบ GPU (CUDA) กำลังใช้งาน RealESRGAN บน CPU ซึ่งจะประมวลผลช้ามาก')
upscaler = RealESRGANer(
    scale=4,
    model_path='weights/RealESRGAN_x4plus.pth',
    model=model,
    device=device
)

# เก็บเฟรมทั้งหมดไว้ในลิสต์

# กำหนดจำนวนเฟรมสูงสุดที่ต้องการประมวลผล

print('กำลังเปิดไลฟ์สตรีมและปรับภาพให้คมชัด...')
start_time = time.time()
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.01)
        continue
    # ลดขนาดเฟรมก่อน upscale เพื่อให้ AI ประมวลผลเร็วขึ้น
    small_frame = cv2.resize(frame, (854, 480), interpolation=cv2.INTER_AREA)
    try:
        sr_frame, _ = upscaler.enhance(small_frame, outscale=2)  # scale=2
        resized_frame = cv2.resize(sr_frame, (1280, 720), interpolation=cv2.INTER_AREA)
        cv2.imshow('Super-Resolution Stream', resized_frame)
        video_writer.write(resized_frame)  # บันทึกเฟรมลงไฟล์
        frame_count += 1
        # ตรวจสอบว่าหน้าต่างถูกปิดด้วยเมาส์ (กากะบาท)
        if cv2.getWindowProperty('Super-Resolution Stream', cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    except Exception as e:
        if frame_count % 50 == 0:
            print(f'ข้ามเฟรมเนื่องจาก error: {e}')
        continue
cap.release()
video_writer.release()
cv2.destroyAllWindows()
end_time = time.time()

elapsed = end_time - start_time
print(f"\nดูสตรีมสดทั้งหมด {frame_count} เฟรม ใช้เวลา {elapsed:.2f} วินาที")
print(f"ไฟล์วิดีโอที่บันทึก: {output_path}")
print(torch.__file__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA GPU found")

# --- แปลงไฟล์ MP4 เป็น HLS (.m3u8 + .ts) ด้วย ffmpeg ---
import subprocess
hls_output = 'output_upscaled.m3u8'
print('กำลังแปลงไฟล์ MP4 เป็น HLS สำหรับเปิดบน live server...')
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-i', output_path,
    '-profile:v', 'baseline',
    '-level', '3.0',
    '-start_number', '0',
    '-hls_time', '4',
    '-hls_list_size', '0',
    '-f', 'hls',
    hls_output
]
try:
    subprocess.run(ffmpeg_cmd, check=True)
    print(f'แปลงสำเร็จ: {hls_output} สามารถนำไปเปิดบน live server/web ได้ทันที')
except Exception as e:
    print(f'แปลง HLS ไม่สำเร็จ: {e}')
