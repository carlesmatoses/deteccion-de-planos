
# ffmpeg -i inputvideo.mp4 -filter_complex "select='gt(scene,0.3)',metadata=print:file=time.txt" -vsync vfr img%03d.png 

from asyncio.windows_events import NULL
import ffmpeg


video_path = "videos/hero.mp4"

(
    ffmpeg
    .input(video_path)
    .filter('select', 'gte(n,{})'.format(0.4))
    .filter('showinfo')
    .output("test.mp4")
    .run(capture_stdout=True)
)