![alt text](C:\tensorflow1\squeezeDet\README\Cover.svg)This repository contains the code accompanying the work titled  **''An investigation of regression as an avenue to find precision-runtime trade-off for object segmentation''**. If you find this work useful, please consider citing the [report](#bibtex).

### Table of Contents

- [Installation](#installation)
- [Main contributions](#contributions)
- [Qualitative Results](#results)
- [Training](#training)
	- [KITTI specific data formatting](#kittispecifictdataformatting)
	- [Cityscape specific data formatting](#cityscapespecificdataformatting)
	- [Training instructions][#traininstructions]
- [Inference](#infer)  	
- [Custom dataset support](#datasetsupport)
- [Contributing guideline](#thirdparty)
- [Maintainers](#team)
- [Bibtex](#bibtex)

### Installation

This installation assumes Anaconda environment is already installed. If not please follow the instructions provided by https://docs.anaconda.com/anaconda/install/.

- Clone this repository.

  ```Shell
  git clone https://github.com/DRealArun/squeezeDet.git
  ```
  Let's call the top level directory of SqueezeDet `$SQDT_ROOT`. 

- Create a new virtual environment.

    ```Shell
  conda create --name squeezeDetOcta python=3.6
  ```
  
- Launch the new environment.

    ```Shell
    # Windows
    activate squeezeDetOcta
    # Linux
    source activate squeezeDetOcta
    ```

- Install the following packages.
  
    ```Shell
    conda install tensorflow-gpu==1.9.0
    conda install -c conda-forge opencv=3.4.2
    conda install -c conda-forge easydict
    conda install -c anaconda pillow
    conda install -c conda-forge imageio
    conda install -c anaconda joblib
    conda install -c anaconda scipy
    # optional
    conda install -c conda-forge jupyterlab
    conda install -c anaconda protobuf
    ```
### Main Contributions

This work builds up on the work **_SqueezeDet:_Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving** by Bichen Wu, Alvin Wan, Forrest Iandola, Peter H. Jin, Kurt Keutzer (UC Berkeley & DeepScale).  

We thank the authors for making the source code openly available. The main contributions of this repository are as follows,

1. ***SqueezeDetOcta*** was developed by augmenting the existing ***SqueezeDet*** object detection network to predict the parameters of irregular octagonal approximations of the instance masks. 

   

   <img src="README\center_offset_bbox.png" width="350"/> <img src="README\center_offset_mask.png" width="350"/> 

   ​		**(a)** *Bounding box approximation*								**(b)** *Octagonal approximation*	

   ​					***Figure 1:*** *Illustrations of different instance masks approximations*

   This involves code changes to,

   1. load instance masks. (Cityscape dataset is chosen for this. But the code can easily be extended for other datasets as described in the [Custom dataset support](#datasetsupport) section.)
   2. approximate the instance masks using irregular octagonal parameterization.
   3. encode these parameters to generate the ground-truth which can be used to train the network.

2. Implementation to continue training from an existing checkpoint.

3. This work uncovered the issues posed by boundary adhering object instances and why they warrant separate handling (refer the report for more information). 

<p align="center">  <img src="README\BoundaryAdhesion.png" width="500"/> </p>

  				***Figure 2:*** *Illustrations of boundary adhering object instances and occluded object instances*

Towards this end, a mechanism for automatic handling of these problematic object instances during network training was proposed. This involves,

  - Introduction of a robust mechanism for automatic identification of problematic image border adhering object instances, compatible with the data-augmentation strategies like random horizontal flipping and horizontal and vertical image translation and image cropping.

  - Introduction of alternate ground-truth encoding/decoding schemes which are better suited for encoding/decoding the decoupled bounding box parameterization. In decoupled bounding box parameterization, each border of the bounding box is independently parameterized i.e., each bounding box is represented by the quartet of *xmin*, *ymin*, *xmax* and *ymax* instead of the usual center-coordinates *(cx, cy)* and width *(w)* and height *(h)*.

  - Introduction of a modified L2 loss function for regression. This loss acts like a normal L2 loss for the object instances which are not in contact with the image boundaries. However, for the problematic image border adhering object instances, it enables only selective learning of the partial untainted parameters.

    

### Qualitative Results



<p align="center"><img src="./README/Stuttgart_bbox.gif" width="400" height="256" /><img src="./README/Stuttgart_octa.gif" width="400" height="256" /></p>

### Training

Currently the repository supports two autonomous driving datasets KITTI and Cityscape but can be extended to custom datasets using the instructions provided in the [Custom dataset support](#datasetsupport).
Before training, download the CNN model pretrained for ImageNet classification.

  ```Shell
# Linux

cd $SQDT_ROOT/data/
# SqueezeNet
wget https://www.dropbox.com/s/fzvtkc42hu3xw47/SqueezeNet.tgz
tar -xzvf SqueezeNet.tgz
# ResNet50 
wget https://www.dropbox.com/s/p65lktictdq011t/ResNet.tgz
tar -xzvf ResNet.tgz
# VGG16
wget https://www.dropbox.com/s/zxd72nj012lzrlf/VGG16.tgz
tar -xzvf VGG16.tgz

# Windows (use the above mentioned links to manually download and untar the weights)
  ```

#### KITTI specific data formatting

- Download KITTI object detection dataset: [images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip) and [labels](http://www.cvlibs.net/download.php?file=data_object_label_2.zip). Put them under `$SQDT_ROOT/data/KITTI/`. Unzip them, then you will get two directories:  `$SQDT_ROOT/data/KITTI/training/` and `$SQDT_ROOT/data/KITTI/testing/`. 

- Split the training data into a training set and a validation set. 

  `trainval.txt` contains indices to all the images in the training data. In the experiments, we randomly split half of indices in `trainval.txt` into `train.txt` to form a training set and rest of them into `val.txt` to form a validation set. For your convenience, we provide a script to split the train-val set automatically. Simply run

    ```Shell
  cd $SQDT_ROOT/data
  
  # Linux
  python kitti_formatting.py --data_path=KITTI
  
  # Windows
  python kitti_formatting.py --data_path=KITTI
    ```

  then you should get the `train.txt` and `val.txt` under `$SQDT_ROOT/data/KITTI/ImageSets`. 

  When the above step is finished, the structure of `$SQDT_ROOT/data/KITTI/` should contain:

  ```Shell
  $SQDT_ROOT/data/KITTI/
                    |->training/
                    |     |-> image_2/00****.png
                    |     L-> label_2/00****.txt
                    |->testing/
                    |     L-> image_2/00****.png
                    L->ImageSets/
                          |-> trainval.txt
                          |-> train.txt
                          L-> val.txt
  ```
#### Cityscape specific data formatting
- Download Cityscape instance segmentation dataset: [images](https://www.cityscapes-dataset.com/file-handling/?packageID=3) and [labels](https://www.cityscapes-dataset.com/file-handling/?packageID=1). Put them under `$SQDT_ROOT/data/Cityscape/`. Unzip them, then you will get two directories:  `$SQDT_ROOT/data/Cityscape/leftImg8bit` and `$SQDT_ROOT/data/Cityscape/gtFine`. 

- Reformat the dataset folder structure. 

  For your convenience, we provide a script to automatically restructure the dataset folder structure. Simply run

    ```Shell
  cd $SQDT_ROOT/data
  
  # Linux
  python cityscape_formatting.py --data_path=Cityscape
  
  # Windows
  python cityscape_formatting.py --data_path=Cityscape
    ```

  then you should get the `train.txt` and `val.txt` and `test.txt` under `$SQDT_ROOT/data/Cityscape/leftImg8bit/ImageSets`.  `train.txt` will contain indices to all the images in the training data.  `val.txt` will contain indices to all the images in the validation data. `test.txt` will contain indices to all the images in the test data. When the above step is finished, the structure of `$SQDT_ROOT/data/Cityscape/` should resemble:

  ```Shell
$SQDT_ROOT/data/Cityscape/
                    |-> leftImg8bit/
                    			|-> train/
                    			|     |-> image_2/***_***_***_leftImg8bit.png
                    			|     |-> instance/***_***_***_gtFine_color.png
                    			|	  |-> instance/***_***_***_gtFine_instanceIds.png
                    			|     |-> instance/***_***_***_gtFine_labelIds.png
                    			|     L-> instance/***_***_***_gtFine_polygons.json
                    			| ...
                    			|-> val/
                    			|     |-> image_2/***_***_***_leftImg8bit.png
                    			|     |-> instance/***_***_***_gtFine_color.png
                    			|	  |-> instance/***_***_***_gtFine_instanceIds.png
                    			|     |-> instance/***_***_***_gtFine_labelIds.png
                    			|     L-> instance/***_***_***_gtFine_polygons.json
                    			| ...
                    			|-> test/
                    			|     |-> image_2/***_***_***_leftImg8bit.png
                    			|     |-> instance/***_***_***_gtFine_color.png
                    			|	  |-> instance/***_***_***_gtFine_instanceIds.png
                    			|     |-> instance/***_***_***_gtFine_labelIds.png
                    			|     L-> instance/***_***_***_gtFine_polygons.json
                    			| ...
                    			L-> ImageSets/
                          			|-> train.txt
                          			|-> val.txt
                          			L-> test.txt
  ```



#### Training instructions

The generalized training command is as follows:

```shell
cd $SQDT_ROOT

python ./src/train.py arguments
```

The available **arguments** and their accepted **values** are as specified in the below tables

**Mandatory training arguments**

| Placeholder              | Accepted value                                               |
| :----------------------- | ------------------------------------------------------------ |
| --dataset=               | [`CITYSCAPE` or `KITTI`]                                     |
| --pretrained_model_path= | OS specific path to the pretrained weights <sub> [squeezenet_v1.1.pkl (for SqueezeDet/SqueezeDetOcta) and squeezenet_v1.0_SR_0.750.pkl (for SqueezeDet+/SqueezeDetOcta+) ]</sub> |
| --data_path=             | OS specific path to the dataset root folder [`$SQDT_ROOT/data/Cityscape/leftImg8bit` or `$SQDT_ROOT/data/KITTI`] |
| --image_set=             | `train`                                                      |
| --train_dir=             | OS specific path to the folder in which logs and checkpoints will be stored |
| --net=                   | [`vgg16`, `resnet50`, `squeezeDet`, `squeezeDet+`]           |
| --summary_step=          | Logging interval in steps                                    |
| --checkpoint_step=       | Checkpoint interval in steps                                 |
| --mask_parameterization= | [`4` or `8`] (*KITTI does not support 8 point parameterization*) |

**Optional training arguments**

| Placeholder                | Accepted value                                               |
| :------------------------- | ------------------------------------------------------------ |
| --eval_valid               | This is a Boolean flag to enable validation set evaluation   |
| --max_steps=               | Maximum number of training steps                             |
| --encoding_type=           | [`asymmetric_linear`, `asymmetric_log`, `normal`]            |
| --log_anchors              | This is a Boolean flag to pair the network with logarithmically extracted anchors. |
| --warm_restart_lr=         | A floating point value to specify initial learning rate.     |
| --bounding_box_checkpoint= | This is a Boolean flag to indicate if the checkpoint in the log folder is for a bounding box predicting network. |
| --only_tune_last_layer     | This is a Boolean flag to indicate the training script to tune only the last layer and keep all the other layer weights fixed. |
| --gpu                      | id of the GPU to be used for training.                       |

The following is an example of a training command to train a SqueezeDetOcta network on Cityscape dataset, with the loss logs being saved every 100 steps and the checkpoints being saved every 500 steps. 

```python
#Linux
python ./src/train.py --dataset=CITYSCAPE --pretrained_model_path=./data/SqueezeNet/squeezenet_v1.1.pkl --data_path=./data/Cityscape/leftImg8bit --image_set=train --train_dir=./logs/train_cityscape_squeezeDetOcta --net=squeezeDet --summary_step=100 --checkpoint_step=500 --mask_parameterization=8

#Windows
python ./src/train.py --dataset=CITYSCAPE --pretrained_model_path=data\SqueezeNet\squeezenet_v1.1.pkl --data_path=data\Cityscape\leftImg8bit --image_set=train --train_dir=logs\train_cityscape8_squeeze_without_val --net=squeezeDet --summary_step=100 --checkpoint_step=500 --mask_parameterization 8
```

Monitor the training process using tensorboard using the command:

```Shell
tensorboard --logdir=$LOG_DIR
```
Here, `$LOG_DIR` is the directory where your logs are dumped, which should be the same as `-train_dir`. 

### Inference

The following checkpoints trained on cityscape are made available.

| Link                                                      | Description                                                  |
| --------------------------------------------------------- | ------------------------------------------------------------ |
| https://1drv.ms/u/s!AjjhUaE_7YtogWDhpk6CWaKjdv0h?e=SefgJF | SqueezeDet with logarithmically extracted anchors and normal encoding. |
| https://1drv.ms/u/s!AjjhUaE_7YtogWF8i7621P1kfCMv?e=1UsLX8 | SqueezeDetOcta with logarithmically extracted anchors and normal encoding. |
| https://1drv.ms/u/s!AjjhUaE_7YtogWJ7HIpuhXXizc2O?e=LjD3dO | SqueezeDet with linearly extracted anchors and anchor-offset linear encoding. |
| https://1drv.ms/u/s!AjjhUaE_7YtogWWkmb4Bd63j_1uY?e=dMOyzg | SqueezeDetOcta with linearly extracted anchors and anchor-offset linear encoding. |
| https://1drv.ms/u/s!AjjhUaE_7YtogWPT6wLieS8oscnd?e=DN6YiD | SqueezeDet with logarithmically extracted anchors and anchor-offset non-linear encoding. |
| https://1drv.ms/u/s!AjjhUaE_7YtogWbK1n5WKcyYEk1c?e=Ge8cEa | SqueezeDetOcta with logarithmically extracted anchors and anchor-offset non-linear encoding. |

<sub> The checkpoints provided by the original repository can be found [here](https://www.dropbox.com/s/a6t3er8f03gdl4z/model_checkpoints.tgz?dl=0). </sub>

- Create a checkpoints folder `$SQDT_ROOT/data/Checkpoints`. 
- Download the zip folder and then unzip it in the folder, The folder structure should resemble,
```Shell
Checkpoints
    |-> checkpoint_folder_1/
    |     	|-> model.ckpt-200000.data-00000-of-00001
    |     	|-> model.ckpt-200000.index
    |	  	L-> model.ckpt-200000
    | ...
    |-> checkpoint_folder_2/
            |-> model.ckpt-200000.data-00000-of-00001
            |-> model.ckpt-200000.index
            L-> model.ckpt-200000
```

- It is always a good practice to convert the checkpoints to a frozen inference graph and then use it for inference. For this reason a utility script is provided which reads the various available checkpoint folders and deduces the parameters to be used to generate the frozen inference graph from the folder names (automatically). Just run the following command to generate inference graphs for all the checkpoints in the `$SQDT_ROOT/data/Checkpoints` folder.


```shell
#Linux
python ./src/inference_graph_for_all.py --train_dir=data/Checkpoints --out_dir=$OUT_DIR

#Windows
python ./src/inference_graph_for_all.py --train_dir=data\Checkpoints --out_dir=$OUT_DIR
```

Here, `$OUT_DIR` is the directory where your inference graph will be written.

- Finally run the inference script to test the model.
```shell
# For the frozen inference graph corresponding to train_4_log_1
python ./src/inference.py --inference_graph=$OUT_DIR\train_4_log_1\frozen_inference_graph.pb --input_path=$INP_DIR --out_dir=$RES_DIR --demo_net=squeezeDet --mask_parameterization_inf=4 --log_anchors_inf --encoding_type_inf=normal --dataset_inf=CITYSCAPE

# For the frozen inference graph corresponding to all_layers_LR_initial_1
python ./src/inference.py --inference_graph=$OUT_DIR\all_layers_LR_initial_1\frozen_inference_graph.pb --input_path=$INP_DIR --out_dir=$RES_DIR --demo_net=squeezeDet --mask_parameterization_inf=8 --log_anchors_inf --encoding_type_inf=normal --dataset_inf=CITYSCAPE

# For the frozen inference graph corresponding to pt_4_lin_lin_anch_1
python ./src/inference.py --inference_graph=$OUT_DIR\pt_4_lin_lin_anch_1\frozen_inference_graph.pb --input_path=$INP_DIR --out_dir=$RES_DIR --demo_net=squeezeDet --mask_parameterization_inf=4 --encoding_type_inf=asymmetric_linear --dataset_inf=CITYSCAPE

# For the frozen inference graph corresponding to pt_8_lin_lin_anch_all_3
python ./src/inference.py --inference_graph=$OUT_DIR\pt_8_lin_lin_anch_all_3\frozen_inference_graph.pb --input_path=$INP_DIR --out_dir=$RES_DIR --demo_net=squeezeDet --mask_parameterization_inf=8 --encoding_type_inf=asymmetric_linear --dataset_inf=CITYSCAPE

# For the frozen inference graph corresponding to pt_4_log_log_anch_3
python ./src/inference.py --inference_graph=$OUT_DIR\pt_4_log_log_anch_3\frozen_inference_graph.pb --input_path=$INP_DIR --out_dir=$RES_DIR --demo_net=squeezeDet --mask_parameterization_inf=4 --log_anchors_inf --encoding_type_inf=asymmetric_log --dataset_inf=CITYSCAPE

# For the frozen inference graph corresponding to pt_8_log_log_anch_all_2
python ./src/inference.py --inference_graph=$OUT_DIR\pt_8_log_log_anch_all_2\frozen_inference_graph.pb --input_path=$INP_DIR --out_dir=$RES_DIR --demo_net=squeezeDet --mask_parameterization_inf=8 --log_anchors_inf --encoding_type_inf=asymmetric_log --dataset_inf=CITYSCAPE
```

Here, 

1. `$INP_DIR` is an OS specific path to an image folder (`./image_dir/00000*.png`) or an video file (`./video_dir/input_1.mp4`).
2. `$RES_DIR` is an OS specific path to an directory into which the processed images/frames will be written.
3. `$INF_GRAPH` is an OS specific path to the frozen inference graph.

### Custom dataset support

Adding support for new datasets is quite simple.  To explain this, consider a dummy dataset. The changes are needed for adding support for this dataset with 2 classes (`class_1` and `class_2`) are as follows,

- File: `$SQDT_ROOT/src/config/config.py` at line number 30 add the following lines.

  ```python
  elif cfg.DATASET == 'CITYSCAPE':
        cfg.CLASS_NAMES = tuple(sorted(('class_1', 'class_2')))
  ```
  
- File: `$SQDT_ROOT/src/config/train.py` modify the line number `168` to.
  ```python
  assert FLAGS.dataset == 'KITTI' or FLAGS.dataset == 'CITYSCAPE' or FLAGS.dataset == 'DUMMY',  'Currently only support KITTI, CITYSCAPE and DUMMY datasets'
  ```
  If the image sizes of the images in the dummy dataset are closer to KITTI image sizes, modify the lines, `178`, `191`, `202` and `213` to,
  ```python
  if FLAGS.dataset == 'KITTI' or FLAGS.dataset == 'DUMMY':
  ```
  If the image sizes of the images in the dummy dataset are closer to Cityscape image sizes, modify the lines, `180`, `193`, `204` and `215` to,
  ```python
  if FLAGS.dataset == 'CITYSCAPE' or FLAGS.dataset == 'DUMMY':
  ```
  At line 237 add the following lines,
  ```python
  elif FLAGS.dataset == 'DUMMY':
      imdb = dummy(FLAGS.image_set, FLAGS.data_path, mc)
      if FLAGS.eval_valid:
          imdb_valid = dummy('val', FLAGS.data_path, mc)
          imdb_valid.mc.DATA_AUGMENTATION = False
  ```
- New File: `$SQDT_ROOT/src/dataset/dummy.py`
 For this dataset, we define a child class names `dummy` in the file `$SQDT_ROOT/src/dataset/dummy.py` which extends the class `input_reader` defined in the `$SQDT_ROOT/src/dataset/input_reader.py` file.

    ```python
    import cv2
    import os 
    ...
    ...

    class dummy(input_reader):
      def __init__(self, image_set, data_path, mc):
        input_reader.__init__(self, 'dummy_'+image_set, mc)
        self._image_set = image_set
        self._data_root_path = data_path
        self._image_path = # Folder to image files
        self._label_path = # Folder to annotation files
        self._classes = self.mc.CLASS_NAMES
        self._class_to_idx = dict(zip(self.classes, range(self.num_classes)))
        self.left_margin = 0
        self.right_margin = 0
        self.top_margin = 0
        self.bottom_margin = 0

        # a list of string indices of images in the directory
        self._image_idx = self._load_image_set_idx() 
        print("Image set chosen: ", self._image_set, "and number of samples: ", len(self._image_idx))
   
     self._rois, self._poly, self._boundary_adhesions = self._load_dummy_annotations()
        self._perm_idx = None
        self._cur_idx = 0
        self._shuffle_image_idx()
        self._eval_tool = None
        
        def _load_image_set_idx(self):
           ''' Function which reads the file names and returns them'''
        
        def _load_cityscape_annotations(self):
           ''' Function which reads and the returns the annotations
           	(bounding boxes or points representing polygons). It
           	also returns the if the boundary adhesion condition 
           	vector for each ground truth annotation.
           '''    
    ```

For more information of these changes please refer the file `$SQDT_ROOT/src/dataset/cityscape.py`

### Contributing guideline

Pull requests to this repository are encouraged. The following guidelines apply.

- For issue creation the following information is mandatory.
  - **Title** : Each title should have the following template [`Issue: short-description-of-issue`]
  - **Comment :** In the comment please describe the issue and detailed steps to reproduce it.
- For pull requests the following guidelines need to be followed.
  - Please provide relevant names to the branches.
  - **Title :** Each title should have the following template [`PurposeTag: short-description-of-pull request`] where `PurposeTag` can have the following values, 
    - `FEATURE` : For new feature implementations.
    - `IMPROVEMENT:` For improvements in the current implementation.
    - `BUG FIX`: For pull-request solving already created Issues.
  - **Comment :** In the comment please describe the purpose of the pull request in detail.

### Maintainers

Currently this repository is maintained by just me.  Would love to share the responsibility with interested developers. If interested please feel free to contact me on [LinkedIn](www.linkedin.com/in/arun-prabhu-0a237074) or by [email][gitecarp@gmail.com].

### Bibtex

```latex
@MastersThesis{2020Prabhu,
  Title                    = {An investigation of regression as an avenue to find precision-runtime trade-off for object segmentation.},
  Author                   = {Prabhu, Arun Rajendra},
  School                   = {Hochschule Bonn-Rhein-Sieg},
  Year                     = {2020},

  Address                  = {Grantham-Allee 20, 53757 St. Augustin, Germany},
  Month                    = {June},
  Note                     = {WS17/18 H-BRS and Fraunhofer IAIS Pl{\"o}ger, Hinkenjann, Eickeler supervising},

  Abstract                 = {The ability to finely segment different instances of various objects in an environment forms a critical tool in the perception tool-box of any autonomous agent. Traditionally instance segmentation is treated as a multi-label pixel-wise classification problem. This formulation has resulted in networks that are capable of producing high-quality instance masks but are extremely slow for real-world usage, especially on platforms with limited computational capabilities. This thesis investigates an alternate regression-based formulation of instance segmentation to achieve a good trade-off between mask precision and run-time. Particularly the instance masks are parameterized and a CNN is trained to regress to these parameters, analogous to bounding box regression performed by an object detection network.

In this investigation, the instance segmentation masks in the Cityscape dataset are approximated using irregular octagons and an existing object detector network (i.e., SqueezeDet) is modified to regresses to the parameters of these octagonal approximations. The resulting network is referred to as SqueezeDetOcta. At the image boundaries, object instances are only partially visible. Due to the convolutional nature of most object detection networks, special handling of the boundary adhering object instances is warranted. However, the current object detection techniques seem to be unaffected by this and handle all the object instances alike. To this end, this work proposes selectively learning only partial, untainted parameters of the bounding box approximation of the boundary adhering object instances. Anchor-based object detection networks like SqueezeDet and YOLOv2 have a discrepancy between the ground-truth encoding/decoding scheme and the coordinate space used for clustering, to generate the prior anchor shapes. To resolve this disagreement, this work proposes clustering in a space defined by two coordinate axes representing the natural log transformations of the width and height of the ground-truth bounding boxes.

When both SqueezeDet and SqueezeDetOcta were trained from scratch, SqueezeDetOcta lagged behind the SqueezeDet network by a massive $\approx$ \textbf{6.19 mAP}. Further analysis revealed that the sparsity of the annotated data was the reason for this lackluster performance of the SqueezeDetOcta network. To mitigate this issue transfer-learning was used to fine-tune the SqueezeDetOcta network starting from the trained weights of the SqueezeDet network. When all the layers of the SqueezeDetOcta were fine-tuned, it outperformed the SqueezeDet network paired with logarithmically extracted anchors by $\approx$ \textbf{0.77 mAP}. In addition to this, the forward pass latencies of both SqueezeDet and SqueezeDetOcta are close to $\approx$ \textbf{19ms}. Boundary adhesion considerations, during training, resulted in an improvement of $\approx$ \textbf{2.62 mAP} of the baseline SqueezeDet network. A SqueezeDet network paired with logarithmically extracted anchors improved the performance of the baseline SqueezeDet network by $\approx$ \textbf{1.85 mAP}.

In summary, this work demonstrates that if given sufficient fine instance annotated data, an existing object detection network can be modified to predict much finer approximations (i.e., irregular octagons) of the instance annotations, whilst having the same forward pass latency as that of the bounding box predicting network. The results justify the merits of logarithmically extracted anchors to boost the performance of any anchor-based object detection network. The results also showed that the special handling of image boundary adhering object instances produces more performant object detectors.}
}
```

