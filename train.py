import os, sys
import json
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss


# Read config file
# if len(sys.argv) != 2:
#     raise SyntaxError("Wrong number of arguments.")
# else:
#     config_fname = sys.argv[1].strip('.py')
#     exec(f"from core.{config_fname} import cfg")
from core.config import cfg

# Read training and validation sets
trainset = Dataset('train')
validset = Dataset('valid')

# Set up some variables
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

# Define input/output tensors for model creation
input_tensor = tf.keras.layers.Input([cfg.YOLO.SIZE, cfg.YOLO.SIZE, 3])
conv_tensors = YOLOv3(input_tensor)
output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor, i)
    output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)

# Create model and optimizer
model = tf.keras.Model(input_tensor, output_tensors)

# Create TF log directory (for tensorboard), cleans it beforehand if already exists
if cfg.TRAIN.DO_TBOARD:
    logdir = f"./runs/{cfg.YOLO.ROOT}/log"
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

# Create copy of config file for safe keeping
out_cfgfile = open(f"./runs/{cfg.YOLO.ROOT}/input_cfg.txt", "w")
json.dump(cfg, out_cfgfile, indent = 4)
out_cfgfile.close()

# If resuming training:
#   1) load previous weights
#   2) adjust global_steps counter
#   3) adjust starting value of learning rate where it left off
if cfg.RESUME.DO_RESUME:
    model.load_weights(f"./runs/{cfg.YOLO.ROOT}/yolov3_epoch{cfg.RESUME.FROM_EPOCH}")
    global_steps.assign_add((cfg.RESUME.FROM_EPOCH + 1) * steps_per_epoch)
    if global_steps < warmup_steps:
        starting_lr = global_steps.numpy() / warmup_steps * cfg.TRAIN.LR_INIT
    elif global_steps < total_steps:
        starting_lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
            (1 + tf.cos((global_steps.numpy() - warmup_steps) / (total_steps - warmup_steps) * np.pi))
        )
    else:
        starting_lr = cfg.TRAIN.LR_END
else:
    starting_lr = 1. / warmup_steps * cfg.TRAIN.LR_INIT

# Create optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=starting_lr)

# Main training function
def train_step(image_data, target):

    if not cfg.TRAIN.BATCH_ONE_BY_ONE:
        with tf.GradientTape() as tape:
            # Predict result
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0
            # Compute losses
            for i in range(3):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]
            total_loss = giou_loss + conf_loss + prob_loss
            # Compute gradient
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    else:
        grads = []
        for ix in range(cfg.TRAIN.BATCH_SIZE):
            with tf.GradientTape() as tape:
                giou_loss = conf_loss = prob_loss = 0
                print(ix)
                # Predict result
                pred_result = model(
                    tf.convert_to_tensor(image_data[[ix], ...]),
                    # training=True, # probably important to keep in mind!
                    training=False,
                )
                # Compute losses
                for i in range(3):
                    conv, pred = pred_result[i*2], pred_result[i*2+1]
                    tgt = (target[i][0][[ix], ...], target[i][1][[ix], ...])
                    loss_items = compute_loss(pred, conv, *tgt, i)
                    giou_loss += loss_items[0] / cfg.TRAIN.BATCH_SIZE
                    conf_loss += loss_items[1] / cfg.TRAIN.BATCH_SIZE
                    prob_loss += loss_items[2] / cfg.TRAIN.BATCH_SIZE
                total_loss = giou_loss + conf_loss + prob_loss
                # Compute gradient (individuals, then sum all)
                grads.append(tape.gradient(total_loss, model.trainable_variables))
                # Alternative gradient computing (cumulative sum during the loop)
                # if ix == 0:
                #     gradients = tape.gradient(total_loss, model.trainable_variables)
                # else:
                #     tmp_grad = tape.gradient(total_loss, model.trainable_variables)
                #     gradients = [
                #         tf.add(gradients[i], tmp_grad[i]) for i in range(len(gradients))
                #     ]
        # Sum gradients
        gradients = [tf.add_n([g[i] for g in grads]) for i in range(len(grads[0]))]

    # Apply gradient
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Print progress on screen (optional) and in a log file
    if cfg.TRAIN.VERBOSE:
        n_epoch = global_steps // steps_per_epoch
        tf.print("=> EPOCH/STEP : %d/%s" % (n_epoch, global_steps.numpy()))
        tf.print("   * learning rate = %e" % optimizer.lr.numpy())
        tf.print("   * giou_loss = %e" % giou_loss)
        tf.print("   * conf_loss = %e" % conf_loss)
        tf.print("   * prob_loss = %e" % prob_loss)
        tf.print("   * total_loss = %e" % total_loss)
    tf.print(
        "%d  %e  %e  %e  %e  %e" % (
            global_steps,
            optimizer.lr.numpy(),
            giou_loss, conf_loss,
            prob_loss, total_loss,
        ),
        output_stream='file://' + os.getcwd() + f'/runs/{cfg.YOLO.ROOT}/log_train.txt',
    )

    # Update learning rate
    global_steps.assign_add(1)
    if global_steps < warmup_steps:
        lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
    elif global_steps < total_steps:
        lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
            (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
        )
    else:
        lr = global_steps / global_steps * cfg.TRAIN.LR_END
    optimizer.lr.assign(lr.numpy())

    # Writing summary data in TF log folder (for tensorboard)
    if cfg.TRAIN.DO_TBOARD:
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()

# Create TF log directory for validation
if cfg.TRAIN.DO_TBOARD:
    if os.path.exists(f'{logdir}_valid'):
        shutil.rmtree(f'{logdir}_valid')
    validate_writer = tf.summary.create_file_writer(f'{logdir}_valid')

# Main validation function
def validate_step(image_data, target):

    # Predict result
    pred_result = model(image_data, training=False)
    giou_loss = conf_loss = prob_loss = 0

    # Compute losses
    for i in range(3):
        conv, pred = pred_result[i*2], pred_result[i*2+1]
        loss_items = compute_loss(pred, conv, *target[i], i)
        giou_loss += loss_items[0]
        conf_loss += loss_items[1]
        prob_loss += loss_items[2]
    total_loss = giou_loss + conf_loss + prob_loss

    # Print validation results in a log file
    tf.print(
        "%d  %e  %e  %e  %e  %e" % (
            global_steps,
            optimizer.lr.numpy(),
            giou_loss, conf_loss,
            prob_loss, total_loss,
        ),
        output_stream='file://' + os.getcwd() + f'/runs/{cfg.YOLO.ROOT}/log_valid.txt',
    )

    # Writing summary data in TF log folder (for tensorboard)
    if cfg.TRAIN.DO_TBOARD:
        with validate_writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("validate_loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("validate_loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("validate_loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("validate_loss/prob_loss", prob_loss, step=global_steps)
        validate_writer.flush()

    return total_loss


# Main loop
n_epochs = cfg.RESUME.EPOCHS if cfg.RESUME.DO_RESUME else cfg.TRAIN.EPOCHS
n_ini_epoch = (cfg.RESUME.FROM_EPOCH + 1) if cfg.RESUME.DO_RESUME else 0
for i in range(n_epochs):
    # Print progress
    print(f"Epoch {i+1} out of {n_epochs}")
    # Training
    for image_data, target in tqdm(trainset, smoothing=1):
        train_step(image_data, target)
    # Validation
    for image_data, target in tqdm(validset, smoothing=1):
        valid_loss = validate_step(image_data, target)
    # Save weights
    model.save_weights(f"./runs/{cfg.YOLO.ROOT}/yolov3_epoch{n_ini_epoch+i}")
