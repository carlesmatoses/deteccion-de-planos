from __future__ import print_function
import os

# Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.scene_manager import write_scene_list
from scenedetect.scene_manager import save_images
# For caching detection metrics and saving/loading to a stats file
from scenedetect.stats_manager import StatsManager

# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector


def find_scenes(video_path):
    # type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    # Construct our SceneManager and pass it our StatsManager.
    scene_manager = SceneManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    scene_manager.add_detector(ContentDetector())

    # We save our stats file to {VIDEO_PATH}.stats.csv.
    stats_file_path = '%s.stats.csv' % video_path

    scene_list = []

    try:
        # If stats file exists, load it.
        if os.path.exists(stats_file_path):
            # Read stats from CSV file opened in read mode:
            with open(stats_file_path, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file)

        # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor(50)

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list()
        # Each scene is a tuple of (start, end) FrameTimecodes.

        print('List of scenes obtained:')
        for i, scene in enumerate(scene_list):
            print(
                'Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                i+1,
                scene[0].get_timecode(), scene[0].get_frames(),
                scene[1].get_timecode(), scene[1].get_frames(),))

        ### Obtener imagenes por escena
        f = open("demofile.csv", "w")
        write_scene_list(f, scene_list, include_cut_list=True, cut_list=None)
        f.close()

        save_images(
            scene_list, 
            video_manager, 
            num_images=1, 
            frame_margin=1, 
            image_extension='jpg', 
            encoder_param=95, 
            image_name_template='scene/$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER',
            output_dir=None,
            downscale_factor=25,
            show_progress=False,
            scale=None,
            height=None, 
            width=None)




        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            base_timecode = video_manager.get_base_timecode()
            with open(stats_file_path, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)

    finally:
        video_manager.release()

    return scene_list


scenes = find_scenes('videos/a.mp4')