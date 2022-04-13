import os, pandas
import numpy as np
# Standard PySceneDetect imports:
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager, write_scene_list, save_images
# For caching detection metrics and saving/loading to a stats file
from scenedetect.stats_manager import StatsManager

# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector
from scenedetect.thirdparty.simpletable import SimpleTableCell, SimpleTableImage
from scenedetect.thirdparty.simpletable import SimpleTableRow, SimpleTable, HTMLPage

def find_scenes(args):
    video_path = args.video_path
    ## type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager() # util para reutilizar los datos y acelerar procesos
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
        video_manager.set_downscale_factor(10)

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
        f = open("%s.DATOS POR ESCENA.csv"% video_path, "w")
        write_scene_list(f, scene_list, include_cut_list=False, cut_list=scene_manager.get_cut_list())
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
            downscale_factor=2,
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


def write_csv(args,scene_list, scene_manager):
    video_path = args.video_path
    
    # FOLDER 
    isExist = os.path.exists("csv/")
    if not isExist:
  
        # Create a new directory because it does not exist 
        os.makedirs("csv/")
        print("The new directory is created!")

    ### Obtener imagenes por escena
    f = open("csv/%s.DATOS POR ESCENA.csv"% os.path.basename(video_path), "w")
    write_scene_list(f, scene_list, include_cut_list=False, cut_list=scene_manager.get_cut_list())
    f.close()



def simple_find_scenes(args):
    video_path = args.video_path
    ## type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager() # util para reutilizar los datos y acelerar procesos
    # Construct our SceneManager and pass it our StatsManager.
    scene_manager = SceneManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    scene_manager.add_detector(ContentDetector())


    scene_list = []

    try:

        # Set downscale factor to improve processing speed.
        video_manager.set_downscale_factor(10)

        # Start video_manager.
        video_manager.start()

        # Perform scene detection on video_manager.
        scene_manager.detect_scenes(frame_source=video_manager)

        # Obtain list of detected scenes.
        scene_list = scene_manager.get_scene_list()
        # Each scene is a tuple of (start, end) FrameTimecodes.
        
        # imagenes para la prediccion (opcional guardarlas)
        save_images(
            scene_list, 
            video_manager, 
            num_images=1, 
            frame_margin=1, 
            image_extension='jpg', 
            encoder_param=95, 
            image_name_template='scene/$VIDEO_NAME/$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER',
            output_dir=None,
            downscale_factor=4,
            show_progress=False,
            scale=None,
            height=None, 
            width=None)
        # imagenes para html
        save_images(
            scene_list, 
            video_manager, 
            num_images=1, 
            frame_margin=1, 
            image_extension='jpg', 
            encoder_param=95, 
            image_name_template='csv/frame/$VIDEO_NAME/$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER',
            output_dir=None,
            downscale_factor=4,
            show_progress=False,
            scale=None,
            height=None, 
            width=None)



    finally:
        video_manager.release()

    return scene_list, scene_manager

def write_scene_list_html(output_html_filename, scene_list, cut_list=None, css=None,
                          css_class='mytable', image_filenames=None, image_width=None,
                          image_height=None,prediction=None):
    """Writes the given list of scenes to an output file handle in html format.

    Arguments:
        output_html_filename: filename of output html file
        scene_list: List of pairs of FrameTimecodes denoting each scene's start/end FrameTimecode.
        cut_list: Optional list of FrameTimecode objects denoting the cut list (i.e. the frames
            in the video that need to be split to generate individual scenes). If not passed,
            the start times of each scene (besides the 0th scene) is used instead.
        css: String containing all the css information for the resulting html page.
        css_class: String containing the named css class
        image_filenames: dict where key i contains a list with n elements (filenames of
            the n saved images from that scene)
        image_width: Optional desired width of images in table in pixels
        image_height: Optional desired height of images in table in pixels
    """
    if not css:
        css = """
        table.mytable {
            font-family: times;
            font-size:12px;
            color:#000000;
            border-width: 1px;
            border-color: #eeeeee;
            border-collapse: collapse;
            background-color: #ffffff;
            width=100%;
            max-width:550px;
            table-layout:fixed;
        }
        table.mytable th {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #eeeeee;
            background-color: #e6eed6;
            color:#000000;
        }
        table.mytable td {
            border-width: 1px;
            padding: 8px;
            border-style: solid;
            border-color: #eeeeee;
        }
        #code {
            display:inline;
            font-family: courier;
            color: #3d9400;
        }
        #string {
            display:inline;
            font-weight: bold;
        }
        """

    # Output Timecode list
    timecode_table = SimpleTable([["Timecode List:"] +
                                  (cut_list if cut_list else
                                   [start.get_timecode() for start, _ in scene_list[1:]])],
                                 css_class=css_class)

    # Output list of scenes
    header_row = ["Scene Number", "Start Frame", "Start Timecode", "Start Time (seconds)",
                  "End Frame", "End Timecode", "End Time (seconds)",
                  "Length (frames)", "Length (timecode)", "Length (seconds)","imagenes","Contenido","Probabilidad"]
    for i, (start, end) in enumerate(scene_list):
        duration = end - start

        row = SimpleTableRow([
            '%d' % (i+1),
            '%d' % start.get_frames(), start.get_timecode(), '%.3f' % start.get_seconds(),
            '%d' % end.get_frames(), end.get_timecode(), '%.3f' % end.get_seconds(),
            '%d' % duration.get_frames(), duration.get_timecode(), '%.3f' % duration.get_seconds()])

        if image_filenames:
            for image in image_filenames[i]:
                row.add_cell(SimpleTableCell(SimpleTableImage(
                    image, width=image_width, height=image_height)))
        # row de prediccion
        if prediction:
            pred_text=""
            prob_text=""
            for pred in prediction[i]:
               pred_text +=pred[1] +"\n"
               prob_text +=str(pred[2]) +"\n"
            row.add_cell(SimpleTableCell(pred_text))
            row.add_cell(SimpleTableCell(prob_text))


        if i == 0:
            scene_table = SimpleTable(rows=[row], header_row=header_row, css_class=css_class)
        else:
            scene_table.add_row(row=row)
    

    # Write html file
    page = HTMLPage()
    page.add_table(timecode_table)
    page.add_table(scene_table)
    page.css = css
    page.save(output_html_filename)
