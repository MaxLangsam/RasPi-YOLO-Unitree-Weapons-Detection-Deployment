# RasPi_Deployment

Training and deploying YOLO object detection models on Raspberry Pi AI Camera or Hailo AI HAT devices.

# Code to download my custom dataset from Roboflow Universe
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR API KEY")
project = rf.workspace("weaponsdetection-fbbzq").project("gun_knife_v2")
version = project.version(4)
dataset = version.download("yolov8")
                

# Prerequisites
## For Hailo accelerator (AI HAT)
### On your PC
You'll need to create an account and install the Hailo Dataflow Compiler and HailoRT (python whl AND .deb if using Ubuntu) <br>
https://hailo.ai/developer-zone/software-downloads/ <br>
git clone and run setup script for hailo model zoo (i've had issues with installing from hailo.ai) <br>
https://github.com/hailo-ai/hailo_model_zoo/<br>
Though I highly recommend using their docker container instead <br>
https://hailo.ai/developer-zone/documentation/hailo-sw-suite-2025-01/?sp_referrer=suite/suite_install.html#docker-installation
### On the RasPi 4/5
Install raspi Hailo software
```
sudo apt install hailo-all
```

See guide for more info <br>
https://www.raspberrypi.com/documentation/computers/ai.html
<br>
## For the AI Camera (Sony IMX500)
### On your PC
If you are using a <b>YOLO8 model</b> Ultralytics onnx exporter can also convert the model to the correct format for the IMX500. <br>
https://docs.ultralytics.com/integrations/sony-imx500/ <br>
<b>If you are using some other model</b> you'll need to install Sony’s Model Compression Toolkit.
```
pip install model_compression_toolkit
```
And use the Pytorch converter tool
```
pip install imx500-converter[pt]
```

And convert/export your model as seen here <br>
https://github.com/sony/model_optimization/blob/main/tutorials/notebooks/mct_features_notebooks/pytorch/example_pytorch_post_training_quantization.ipynb <br>
I have not yet tested this for other model types (aka YOLO11) #TODO
### On the RasPi
Install the RasPi IMX tools (you only need to run imx500-all)
```
sudo apt install imx500-all
```

## Demos
Raspberry Pi's picamera2 has the functionality and demo Python scripts that you are use once you have done the final conversions <br>
Clone picamera2 repository, install it:<br>

```
git clone -b next https://github.com/raspberrypi/picamera2
cd picamera2
pip install -e .  --break-system-packages
```

# Training/Exporting 
## Dataset Preparation

Organize your images in the following structure:
```
data_directory/
├── images/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image1.jpg
│       └── image2.jpg
├── labels/
│   ├── class1/
│   │   ├── image1.txt
│   │   └── image2.txt
│   └── class2/
│       ├── image1.txt
│       └── image2.txt
└── classes.json
```


Create train/test/validation splits:
```
python create_data_splits.py --data_dir /path/to/data_directory
```

For ONNX export configuration:
```
python create_data_splits.py --data_dir /path/to/data_directory --onnx_config
```

## Training

Train a new model:
```
python yolo_train.py \
    --config /path/to/dataset_config.yaml \
    --init_model yolov8n.pt \
    --name my_model \
    --epochs 20 \
    --device 0
```

Resume training:
```
python yolo_train.py \
    --config /path/to/dataset_config.yaml \
    --name my_model \
    --resume_training
```

## Model ONNX Export

Export to ONNX format:
```
python yolo_train.py \
    --init_model /path/to/trained_model.pt \
    --export_format onnx \
    --export_config /path/to/onnx_config.yaml \
    --export_only
```

Export for IMX500 quantization:
```
python yolo_train.py \
    --init_model /path/to/trained_model.pt \
    --export_format imx \
    --export_only \
    --int8_weights
```

## Calibration Data Generation

Generate calibration data for Hailo converter:
```
python hailo_calibration_data.py \
    --data_dir /path/to/data_directory \
    --target_dir calib \
    --image_size 640 \
    --num_images 1024
```

# Convert the models
## AI Camera (IMX500)
Copy the `packerOut.zip` file and `labels.txt` file from your PC <b>to the RasPi</b> and convert it to the RPK file format <br>
```
imx500-package -i <path to packerOut.zip> -o <output folder>
```
For testing you can connect the IMX500 to the Raspberry Pi and use picamera2 IMX500 demo script!

```
cd picamera2/examples/imx500/
python imx500_object_detection_demo.py --model <path to network.rpk> --labels <path to labels.txt> --fps 25 --bbox-normalization --ignore-dash-labels --bbox-order xy
```
 
## AI HAT (HAILO)
### On your PC
You'll need to perform the conversion, make sure you have run `hailo_calibration_data.py` to create the calib data (`--calib-path`). <br>
Once the YOLO model has finished training (in previous step) the best model will be exported to the ONNX format with default params, 
Use this file to compile `--ckpt`. 
Point `--yaml` to the network config yaml in the hail_model_zoo repo, aka `hailo_model_zoo/hailo_model_zoo/cfg/networks/yolov8n.yaml`.  <br>
Set the number of classes for your model (`--classes`). <br>
Set the `--hw-arch` to your Hailo accelerator type, Hailo 8 (26 TOPS): `hailo8` or Hailo 8L (13 TOPS) `hailo8l`
```
hailomz compile --ckpt <yolo.onnx> --calib-path /path/to/calibration/imgs/dir/ --yaml path/to/yolov8n.yaml --classes <number of classes> --hw-arch hailo8
```

### On the RasPi 4/5
Copy the `yolo.hef` file and `labels.txt` file from your PC <b>to the RasPi</b>, connect a camera, and run the Hailo picamera2 demo script!
```
cd picamera2/examples/hailo/
python detect.py --model <path to yolo.hef> --labels <path to labels.txt>
```
