# On the sensitivity of pose estimation neural networks: rotation parameterizations, Lipschitz constants, and provable bounds
---

This is the code for the paper:

[**On the sensitivity of pose estimation neural networks: rotation parameterizations, Lipschitz constants, and provable bounds**](https://!!!)

Trevor Avant & Kristi A. Morgansen


## dependencies

* Python 3 (we used version 3.8.12), Pytorch (we used version 1.10.2)

* the following non-standard python packages: torchvision (0.11.3), torchsummary (1.5.1), cv2 (4.5.5), numpy (1.22.2), matplotlib (3.5.1), transforms3d (0.3.1), imageio (2.15.0)

* optional: if you want to generate renders of a new object (i.e., an object other than the soup can) and re-train the network, you will need to install Blender


## Monte Carlo checks on mathematical results (`math_verification/` directory)

We verified many of the mathematical results in this paper using a random sampling approach:

* check the rotational distance formula: `rotational_distance_formula_*.py`

* check of the distance ratio constant: `distance_ratio_constant_*.py`

* check various identities: `identity_*.py`

* check Lemma 11 and Theorem 12: `proof_lemma_11.py` and `proof_theorem_12.py`

* check that the triangle inequality holds for the rotational distance function: `triangle_inequality_rotational_distance.py`


## pose network with provable bounds (`network/` directory)

In Section 8, we developed a pose estimation network with provable sensitivity bounds. The code for this network is in the `network/` directory. We trained this network on the "soup can" object, and our pre-trained network can be used by downloading the checkpoint file from [this link](https://drive.google.com/file/d/10G4NwHPUo_O8YV7FH06R6zzbROW93NyT/view?usp=sharing), and then placing that file in `network/data/checkpoints/`. Additionally, if you would like to use an object other than the soup can, the `network/` directory contains code to generate synthetic images of an object using Blender (which requires a `.blend` file) and code to retrain the network.

### analysis of the trained network

The `python analyze_network.py` script calculates several analytical properties of the network:

* positional and rotational Lipschitz bounds

* positional and rotational errors for both train and test data

* the percentage of exponential coordinates with angles greater than pi for both train and test data

Additionally, the `visualize_test_data.py` script allows you to visualize how accurate the network's predictions are by displaying images from the test data with estimates of the object pose overlayed on the image.

### training/test data generation and network training (optional)

1. Obtain random background images to superimpose the renders on. We used images from the [SUN397 dataset](https://vision.princeton.edu/projects/2010/SUN/) for training data and images from the [2017 COCO dataset](https://cocodataset.org/#download) for test data. The images in these datasets have different sizes, but you can scale and crop them to a particular size by running the `image_scripts.py` script. Note you will have to modify the `bkgd_dir` variable in the `dirs.py` file to reference a directory on your computer.

2. generate the renders of the object of interest: `python generate_renders.py`

3. generate training and test data by superimposing the renders on background images: `python train_and_test_data.py`

4. train the network: `python run_train.py`

Note: If you wish to use an object other than the soup can, you need a `.blend` file of the object, and you need to generate the coordinates of a bounding box of the boject which can be done via the `generate_bounding_box.py` Blender/Python script.
