import ffmpeg
video_path = "videos/hero.mp4"
(
    ffmpeg
    .input(video_path)
    .filter('select', 'gte(n,{})'.format(0.3))
    .filter('metadata','print',file='time.txt')
    .output('img%03d.png', vframes=1, format='image2', vcodec='mjpeg')
    .run(capture_stdout=True)
)