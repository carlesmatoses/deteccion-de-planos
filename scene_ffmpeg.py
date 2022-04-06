
# ffmpeg -i inputvideo.mp4 -filter_complex "select='gt(scene,0.3)',metadata=print:file=time.txt" -vsync vfr img%03d.png 

import ffmpeg


video_path = "videos/hero.mp4"

out, _ = (
    ffmpeg
    .input(video_path)
    .filter('select', 'gte(n,{})'.format(0.3),metadata='print:file=time.txt')
    .output('img%03d.png', vframes=1, format='image2', vcodec='mjpeg')
    .view(filename='filter_graph')
    .run(capture_stdout=True)
)