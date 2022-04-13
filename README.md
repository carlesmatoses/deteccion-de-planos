# deteccion-de-planos
Detectar y separar un video por planos y reconocer el contenido.

Llamando a app.py desde el cmd puedes separar un video por escenas junto a datos de estas escenas.

Se pueden emplear diferentes modelos de deep learning como inception o vgg16.

parametros de entrada:
- ```--video_path, -v ```: path to the video
- ```--csv, -csv ```: csv name
- ```--images, -im ```: save images high_res
- ```--prediction_number, -pn ```: number of predictions from most probable to less
- ```--model, -m ```: name of pre-trained network to use
- ```--output, -out ```: Folder where images and html will be stored

El único parametro necesario es el video, los otros se definiran en funcion de este 

## Anaconda
crear entorno en anaconda prompt con:
```conda env create -n m1 -f environment.yaml```

## Ejemplos y uso 

```python app.py -v video_path.mp4 -csv custom_name  -im False -pn 3 -m resnet -out None```
o
```python app.py -v video_path.mp4```

La salida se guardara en una carpeta con el nombre del video por defecto o con el nombre de ```-out``` si este es declarado en la llamada.

## Salida
Se generarán: 
- Un csv
- Un html con imagenes y datos adicionales, 
- Carpeta con imagenes low_res para el html
- carpeta con imagenes high_res si ```--images, -im ``` == True

[Ejemplo de output con un video](fig/NG.html)