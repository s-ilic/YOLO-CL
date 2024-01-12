import os, sys
import cv2
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.yolov3 import YOLOv3, decode


# Read config file
# if len(sys.argv) != 2:
#     raise SyntaxError("Wrong number of arguments.")
# else:
#     config_fname = sys.argv[1].strip('.py')
#     exec(f"from core.{config_fname} import cfg")
from core.config import cfg

# Settings
ix_start = cfg.TEST.IX_START
reso = cfg.TEST.RESO
pix_size = cfg.TEST.PIX_SIZE
pad_size = cfg.TEST.PAD_SIZE

# Prep for real images/galaxies
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

# Path to fits files
pathData="/home/users/ilic/ML/SDSS_fits_data/"

# Read data in fits files
hdul1 = fits.open(pathData+'redmapper_dr8_public_v6.3_catalog.fits')
hdul2 = fits.open(pathData+'redmapper_dr8_public_v6.3_members.fits')
clu_full_data = hdul1[1].data
mem_full_data = hdul2[1].data
n_clu = len(clu_full_data)

# Set up some variables
INPUT_SIZE   = cfg.YOLO.SIZE
NUM_CLASS    = len(utils.read_class_names(cfg.YOLO.CLASSES))
CLASSES      = utils.read_class_names(cfg.YOLO.CLASSES)

# Set up some paths
mAP_dir_path = f'./runs/{cfg.YOLO.ROOT}/mAP_{cfg.TEST.OUTPUT_ROOT}'
predicted_dir_path = f'./runs/{cfg.YOLO.ROOT}/mAP_{cfg.TEST.OUTPUT_ROOT}/predicted'
ground_truth_dir_path = f'./runs/{cfg.YOLO.ROOT}/mAP_{cfg.TEST.OUTPUT_ROOT}/ground-truth'
detected_image_path = f'./runs/{cfg.YOLO.ROOT}/detect_{cfg.TEST.OUTPUT_ROOT}'

# Clean output folders if needed
if ix_start == 0:
    if os.path.exists(mAP_dir_path):
        shutil.rmtree(mAP_dir_path)
    if os.path.exists(predicted_dir_path):
        shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path):
        shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(detected_image_path):
        shutil.rmtree(detected_image_path)
    os.mkdir(mAP_dir_path)
    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(detected_image_path)

# Build model
input_layer  = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
feature_maps = YOLOv3(input_layer)
bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)
model = tf.keras.Model(input_layer, bbox_tensors)
model.load_weights(cfg.TEST.WEIGHTS_FNAME) #loads the weights from the training

# Main loop ~: for each image, runs the network and traces boxes
with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
    for num, line in enumerate(annotation_file):
        if num >= ix_start:
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt=[]
                classes_gt=[]
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')

            # Predict Process
            image_size = image.shape[:2]
            image_data = utils.image_preprocess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            # pred_bbox = model.predict(image_data)
            pred_bbox = model(image_data, training=False)
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)
            bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
            bboxes = utils.nms(
                bboxes,
                cfg.TEST.IOU_THRESHOLD,
                method=cfg.TEST.IOU_METHOD,
                sigma=cfg.TEST.SIGMA,
            )

            # # Draw boxes and galaxies in image
            # if detected_image_path is not None:
            #     # Draw truth
            #     tmp_bboxes = []
            #     for bb in bboxes_gt:
            #         tmp_bb = [1,1,1,1,1,1]
            #         tmp_bb[:4] = bb
            #         tmp_bboxes.append(tmp_bb)
            #     image = utils.draw_bbox(image, tmp_bboxes, classes=['cluster','truth'])
            #     image = utils.draw_bbox(image, bboxes)
            #     # Draw member galaxies
            #     clus_id = int(image_name.split('.')[0])
            #     g1 = np.where(clu_full_data['ID'] == clus_id)[0]
            #     g2 = np.where(mem_full_data['ID'] == clus_id)[0]
            #     c = SkyCoord(
            #         ra=clu_full_data['RA'][g1]*u.degree,
            #         dec=clu_full_data['DEC'][g1]*u.degree,
            #         frame='icrs',
            #     )
            #     c2 = SkyCoord(
            #         ra=mem_full_data['RA'][g2]*u.degree,
            #         dec=mem_full_data['DEC'][g2]*u.degree,
            #         frame='icrs',
            #     )
            #     off = c.spherical_offsets_to(c2)
            #     xs = np.round(pix_size - off[0].arcsec / reso * 1.01).astype('int')
            #     ys = np.round(pix_size - off[1].arcsec / reso * 1.01).astype('int')
            #     for x, y in zip(xs, ys):
            #         image = cv2.circle(image, (x,y), 5, (0,0,255),3)
            #     # Save image
            #     cv2.imwrite(detected_image_path+'/'+image_name, image)

            image = utils.draw_bbox(image, bboxes)
            cv2.imwrite(detected_image_path+'/'+image_name, image)


            # Output prediction results
            with open(predict_result_path, 'w') as f:
                for bbox in bboxes:
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    class_name = CLASSES[class_ind]
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())

