# Final Year Project: Integration of Delta Robit and advanced Computer Vision


## OBJECTIVE
Using Top-down method to the problem of object detection and sorting.  

## Installation

Clone the repository locally:

```shell
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
```

Create the conda env. The code requires `python>=3.7`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

```shell
conda create -n FastSAM python=3.9
conda activate FastSAM
```

Install the packages:

```shell
cd FastSAM
pip install -r requirements.txt
```

Install CLIP:

```shell
pip install git+https://github.com/openai/CLIP.git
```

## <a name="GettingStarted"></a> Getting Started


First download a [model checkpoint](#model-checkpoints).

Then, you can run the scripts to try the everything mode and three prompt modes.

```shell
# Everything mode
python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg
```

```shell
# Text prompt
python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg  --text_prompt "the yellow dog"
```

```shell
# Box prompt (xywh)
python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --box_prompt "[[570,200,230,400]]"
```

```shell
# Points prompt
python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg  --point_prompt "[[520,360],[620,300]]" --point_label "[1,0]"
```

You can use the following code to generate all masks, make mask selection based on prompts, and visualize the results.
```shell
from fastsam import FastSAM, FastSAMPrompt

model = FastSAM('./weights/FastSAM.pt')
IMAGE_PATH = './images/dogs.jpg'
DEVICE = 'cpu'
everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# everything prompt
ann = prompt_process.everything_prompt()

# bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
ann = prompt_process.box_prompt(bbox=[[200, 200, 300, 300]])

# text prompt
ann = prompt_process.text_prompt(text='a photo of a dog')

# point prompt
# points default [[0,0]] [[x1,y1],[x2,y2]]
# point_label default [0] [1,0] 0:background, 1:foreground
ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])

prompt_process.plot(annotations=ann,output_path='./output/dog.jpg',)
```
