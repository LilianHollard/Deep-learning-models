#https://sannaperzon.medium.com/yolov3-implementation-with-training-setup-from-scratch-30ecb9751cb0

""" 
Information about architecture config:
Tuple is structured by and signifies a convolutional block (filters, kernel_size, stride) 
Every convolutional layer is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats. 
"S" is for a scale prediction block and computing the yolo loss
"U" is for upsampling the feature map
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    # first route from the end of the previous block
    (512, 3, 2),
    ["B", 8], 
    # second route from the end of the previous block
    (1024, 3, 2),
    ["B", 4],
    # until here is YOLO-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

nano_scale = 0.33

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
] 

DATASET = " /notebooks/lhollard/datasets/PASCAL_VOC/"
IMG_DIR = DATASET + "iamges/"
LABEL_DIR  = DATASET + "labels/"