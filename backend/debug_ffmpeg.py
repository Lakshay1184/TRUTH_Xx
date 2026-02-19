
import imageio_ffmpeg
import subprocess
import os

try:
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"FFmpeg exe: {exe}")

    # Create dummy video
    if not os.path.exists("test.mp4"):
        print("Creating test.mp4...")
        subprocess.run([exe, '-f', 'lavfi', '-i', 'testsrc=duration=5:size=1280x720:rate=30', '-y', 'test.mp4'], check=True, timeout=30)
    
    # Run ffmpeg -i
    print("Running probe...")
    res = subprocess.run([exe, '-i', 'test.mp4', '-hide_banner'], capture_output=True, text=True, encoding='utf-8', errors='replace')
    
    with open('ffmpeg_stderr.txt', 'w', encoding='utf-8') as f:
        f.write(res.stderr)
    
    print("STDERR captured to ffmpeg_stderr.txt")
    print("-" * 20)
    print(res.stderr)
    print("-" * 20)

except Exception as e:
    print(f"Error: {e}")
