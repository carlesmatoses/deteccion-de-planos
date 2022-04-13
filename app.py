import argparse, sys, shutil, os
import numpy as np
import funciones, funciones_tensorflow


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import Xception # TensorFlow ONLY
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19


# ARGUMENTOS
parser=argparse.ArgumentParser()
parser.add_argument('--video_path','-v', help='path to the video', type=str)
parser.add_argument('--csv', help='csv name', type=bool, default=True)
parser.add_argument('--images',"-im", help='save images', type=bool, default=False)
parser.add_argument("--model","-m", type=str, default="vgg16",
	help="name of pre-trained network to use")
args=parser.parse_args()




# VARIABLES 
video_name = str(os.path.splitext(os.path.basename(args.video_path))[0])

MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet": ResNet50
}

# COMPROBACION DE INPUTS 
if not os.path.exists(args.video_path) or  os.path.exists(args.video_path)==None:
    raise ValueError("video path not valid")

# esnure a valid model name was supplied via command line argument
if args.model not in MODELS.keys():
	raise AssertionError("The --model command line argument should "
		"be a key in the `MODELS` dictionary")

# DETECCION POR ESCENAS Y PREDICCION
scene_list, scene_manager = funciones.simple_find_scenes(args) # variables para guardar los datos
# prediction = funciones_tensorflow.inception_predict(f"scene/{video_name}",pred_num=3)

prediction = funciones_tensorflow.model_predict(args,MODELS,f"scene/{video_name}",pred_num=1)

# DICCIONARIOS
img_list = os.listdir(f"csv/frame/{video_name}")
for i in range(len(img_list)):
    img_list[i] = [os.path.join(f"csv/frame/{video_name}/",img_list[i])]
img_dic = dict(zip(np.arange(0,len(img_list)), img_list))

prediction_dic = dict(zip(np.arange(0,len(prediction)), prediction))

# CSS 
funciones.write_csv(args,scene_list, scene_manager)
# HTML
funciones.write_scene_list_html(
                                output_html_filename=f"{video_name}.html",
                                scene_list=scene_list,
                                cut_list=scene_manager.get_cut_list(),
                                image_filenames=img_dic,
                                prediction=prediction_dic, 
                                )



if args.images == False:
    try:
        shutil.rmtree("scene/" + video_name)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
