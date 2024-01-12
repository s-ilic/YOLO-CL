from easydict import EasyDict as edict


__C                           = edict()
cfg                           = __C


####################
### Main options ###
####################

__C.YOLO                      = edict()

### Choose size of the YOLO network (first layer)
__C.YOLO.SIZE = 512

### Choose name of output folder (code will create it in "./runs")
__C.YOLO.ROOT = "2048x2048_ds2_0p396_pad50"
# __C.YOLO.ROOT = "2048x2048_ds4_0p396_pad50"
# __C.YOLO.ROOT = "2048x2048_ds4_0p396_pad50_zcut0p3"
# __C.YOLO.ROOT = "test2"
# __C.YOLO.ROOT = "2048x2048_ds4_0p396_pad50_zbins_ovl"
# __C.YOLO.ROOT = "2016x2016_ds1_0p396_nopad"
# __C.YOLO.ROOT = "2048x0p396_ds1_mb32_nopad_A"
# __C.YOLO.ROOT = "2048x0p396_ds1_mb32_nopad_B"
# __C.YOLO.ROOT = "2048x0p792_ds1_mb32_nopad_A"

### File containing the YOLO classes names
__C.YOLO.CLASSES              = "./runs/clusters.names"
# __C.YOLO.CLASSES              = "./runs/clusters_zbins.names"
# __C.YOLO.CLASSES              = "./runs/clusters_zbins_ovl.names"

### File containing the YOLO baseline anchors
### NOTES: After doing some clustering studies on ground truth labels, it turns out
### that most bounding boxes have certain height-width ratios. So instead of directly
### predicting a bounding box, YOLOv2 (and v3) predict off-sets from a predetermined
### set of boxes with particular height-width ratios - those predetermined set of
### boxes are the anchor boxes.
__C.YOLO.ANCHORS              = "./runs/baseline_anchors.txt"
# __C.YOLO.ANCHORS              = "./runs/clusters_anchors_2048_A.txt"
# __C.YOLO.ANCHORS              = "./runs/clusters_anchors_1024.txt"
# __C.YOLO.ANCHORS              = "./runs/clusters_anchors_512.txt"

### List of strides == integer factors by which the input images are reduced
### within the YOLO network when performing the multiscale detection
### NB: Factors are applied to TRAIN.INPUT_SIZE, not the "real" image size
__C.YOLO.STRIDES              = [8, 16, 32]

### Self explanatory
__C.YOLO.ANCHOR_PER_SCALE     = 3

### "Intersection Over Union" loss threshold
### From https://stats.stackexchange.com/a/384201 :
### "First, one predicted box is assigned to the ground truth based on which predicted
### box has the (highest?) IoU. Then you have all the other predicted boxes that may not
### have had the highest IoU but do have an IoU over 0.5 with the object. These
### predicted boxes are not assigned to a ground truth but from what I understand, they
### are not included in the loss function (i.e. the last section for no object). Only
### predicted boxes which have an IoU of less than 0.5 with any object are considered in
### the no object loss."
__C.YOLO.IOU_LOSS_THRESH      = 0.5


########################
### Training options ###
########################

__C.TRAIN                     = edict()

### Verbose mode (print losses in terminal during training)
__C.TRAIN.VERBOSE             = True

### Create Tensorboard log files during training
__C.TRAIN.DO_TBOARD           = False

### Path to text file "pointing" to the training set of images; each line of the file
### should contain the full path to an image, followed by the list of bounding boxes
### (+ class number) for all objects in said image
__C.TRAIN.ANNOT_PATH          = "./runs/%s/train.txt" % __C.YOLO.ROOT
# __C.TRAIN.ANNOT_PATH          = "./runs/%s/train_crop.txt" % __C.YOLO.ROOT

# Number of training epochs
__C.TRAIN.EPOCHS              = 60

### How many training images per batch (careful, can saturate GPU memory quickly)
# __C.TRAIN.BATCH_SIZE          = 32
# __C.TRAIN.BATCH_SIZE          = 16
# __C.TRAIN.BATCH_SIZE          = 8
# __C.TRAIN.BATCH_SIZE          = 4
# __C.TRAIN.BATCH_SIZE          = 2
__C.TRAIN.BATCH_SIZE          = 1

### Alternative method of mini-batch to accomodate large images
# __C.TRAIN.BATCH_ONE_BY_ONE    = True
__C.TRAIN.BATCH_ONE_BY_ONE    = False

### List of image sizes which the program randomly picks at training time, and into
### which the input images are converted into
# __C.TRAIN.INPUT_SIZE          = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
# __C.TRAIN.INPUT_SIZE          = [512]
__C.TRAIN.INPUT_SIZE          = [1024]
# __C.TRAIN.INPUT_SIZE          = [2048]

### Choose whether to perform data augmentation (namely, horizontal flip, crop, and
### translation) on training images
__C.TRAIN.DATA_AUG            = True
# __C.TRAIN.DATA_AUG            = False

### Initial and final learning rate, and number of "warmup epochs"
### During the warmup, the learning rate "lr" goes up as:
###    lr = X / WARMUP_EPOCHS * LR_INIT
### and afterwards descends as:
###    lr = LR_END + 0.5 * (LR_INIT - LR_END) *
####      [1 + cos(pi * (X - WARMUP_EPOCHS)/(EPOCHS - WARMUP_EPOCHS))]
#### where X = current_step / steps_per_epoch
###
# __C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_INIT             = 1e-4  # intial learning rate
__C.TRAIN.LR_END              = 1e-6 # final learning rate
# __C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.WARMUP_EPOCHS       = 4
# __C.TRAIN.WARMUP_EPOCHS       = 8


####################
### Test options ###
####################

__C.TEST                      = edict()

### Path to weights file to use for testing
__C.TEST.WEIGHTS_FNAME = "./runs/%s/yolov3_epoch%s" % (cfg.YOLO.ROOT, ntl)int(sys.argv[1]) # which save file to load

### Same definitions as for TRAIN variables, but for test images
__C.TEST.ANNOT_PATH           = "./runs/%s/valid.txt" % __C.YOLO.ROOT
__C.TEST.BATCH_SIZE           = 1 # always keep it like this
# __C.TEST.INPUT_SIZE           = 512
__C.TEST.INPUT_SIZE           = 1024
# __C.TEST.INPUT_SIZE           = 2048
__C.TEST.DATA_AUG             = False

### Additional test images settings required (for drawing, notably)
# Image settings
__C.TEST.IX_START = 0           # which test image to start from (0 = from beginning)
__C.TEST.RESO = 0.396127        # arcsec/pix
__C.TEST.PIX_SIZE = 1024        # image half side size in pixels
__C.TEST.PAD_SIZE = 50          # padding size in pixels


### Discard predicted boxes whose score is below this threshold
__C.TEST.SCORE_THRESHOLD      = 0.3

### Which method for "pruning" detected boxes
### (see nms function in core/utils.py)
__C.TEST.IOU_METHOD        = 'nms'
# __C.TEST.IOU_METHOD        = 'soft-nms'

### Sigma setting for 'soft-nms' method
### (see nms function in core/utils.py)
### NB: setting ignored if TEST.IOU_METHOD = 'nms'
__C.TEST.SIGMA     = 0.3

### Discard boxes that have IOU higher than this threshold with a box of higher score
### (see nms function in core/utils.py)
### NB: setting ignored if TEST.IOU_METHOD = 'soft-nms'
__C.TEST.IOU_THRESHOLD        = 0.45


######################
### Resume options ###
######################

__C.RESUME                    = edict()

### Resume training from a previous run (don't touch previous settings, they are needed)
__C.RESUME.DO_RESUME          = False

### End of which previous epoch to use as starting point (0-indexed)
__C.RESUME.FROM_EPOCH         = 29

### How many more epochs of training requested
__C.RESUME.EPOCHS         = 1000
