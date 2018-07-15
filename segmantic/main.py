#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/7/13 2:08
# author   : QLMX
from __future__ import print_function

import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import os, sys
import random
from PIL import Image

import helpers
import utils
from config import cfg
from tools import buildNetwork

sys.path.append("preprocess")
import dataset

def train():
    if cfg.class_balancing:
        print("Computing class weights for trainlabel ...")
        class_weights = utils.compute_class_weights(labels_dir=train_output_names, label_values=label_values)
        weights = tf.reduce_sum(class_weights * net_output, axis=-1)
        unweighted_loss = None
        unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output)
        losses = unweighted_loss * class_weights
    else:
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output)
    loss = tf.reduce_mean(losses)

    opt = tf.train.AdamOptimizer(cfg.lr).minimize(loss, var_list=[var for var in tf.trainable_variables()])


    sess.run(tf.global_variables_initializer())
    utils.count_params()

    # If a pre-trained ResNet is required, load the weights.
    # This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
    if init_fn is not None:
        init_fn(sess)

    avg_scores_per_epoch = []
    avg_loss_per_epoch = []

    # Which validation images do we want
    val_indices = []
    num_vals = min(cfg.num_val_images, len(val_input_names))

    # Set random seed to make sure models are validated on the same validation images.
    # So you can compare the results of different models more intuitively.
    random.seed(16)
    val_indices = random.sample(range(0, len(val_input_names)), num_vals)

    # Do the training here
    for epoch in range(0, cfg.num_epochs):
        current_losses = []
        cnt = 0

        # Equivalent to shuffling
        id_list = np.random.permutation(len(train_input_names))

        num_iters = int(np.floor(len(id_list) / cfg.batch_size))
        st = time.time()
        epoch_st = time.time()

        for i in range(num_iters):
            # st=time.time()
            input_image_batch = []
            output_image_batch = []

            # Collect a batch of images
            for j in range(cfg.batch_size):
                index = i * cfg.batch_size + j
                id = id_list[index]
                input_image = dataset.load_image(train_input_names[id])
                output_image = dataset.load_image(train_output_names[id])

                with tf.device('/cpu:0'):
                    input_image, output_image = dataset.data_augmentation(input_image, output_image)

                    # Prep the data. Make sure the labels are in one-hot format
                    input_image = np.float32(input_image) / 255.0
                    output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))

                    input_image_batch.append(np.expand_dims(input_image, axis=0))
                    output_image_batch.append(np.expand_dims(output_image, axis=0))

            # ***** THIS CAUSES A MEMORY LEAK AS NEW TENSORS KEEP GETTING CREATED *****
            # input_image = tf.image.crop_to_bounding_box(input_image, offset_height=0, offset_width=0,
            #                                               target_height=args.crop_height, target_width=args.crop_width).eval(session=sess)
            # output_image = tf.image.crop_to_bounding_box(output_image, offset_height=0, offset_width=0,
            #                                               target_height=args.crop_height, target_width=args.crop_width).eval(session=sess)
            # ***** THIS CAUSES A MEMORY LEAK AS NEW TENSORS KEEP GETTING CREATED *****

            # memory()

            if cfg.batch_size == 1:
                input_image_batch = input_image_batch[0]
                output_image_batch = output_image_batch[0]
            else:
                input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
                output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

            # Do the training
            _, current = sess.run([opt, loss], feed_dict={net_input: input_image_batch, net_output: output_image_batch})
            current_losses.append(current)
            cnt = cnt + cfg.batch_size
            if cnt % 20 == 0:
                string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f" % (epoch, cnt, current, time.time() - st)
                utils.LOG(string_print)
                st = time.time()

        mean_loss = np.mean(current_losses)
        avg_loss_per_epoch.append(mean_loss)

        # Create directories if needed
        if not os.path.isdir(cfg.base_dir + "%s/%s/%04d" % ("checkpoints", cfg.model, epoch)):
            os.makedirs(cfg.base_dir + "%s/%s/%04d" % ("checkpoints", cfg.model, epoch))

        # Save latest checkpoint to same file name
        print("Saving latest checkpoint")
        saver.save(sess, model_checkpoint_name)

        if val_indices != 0 and epoch % cfg.checkpoint_step == 0:
            print("Saving checkpoint for this epoch")
            saver.save(sess, cfg.base_dir + "%s/%s/%04d/model.ckpt" % ("checkpoints", cfg.model, epoch))

        if epoch % cfg.validation_step == 0:
            print("Performing validation")
            target = open(cfg.base_dir + "%s/%s/%04d/val_scores.csv" % ("checkpoints", cfg.model, epoch), 'w')
            target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))

            scores_list = []
            class_scores_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            iou_list = []

            # Do the validation on a small set of validation images
            for ind in val_indices:

                input_image = np.expand_dims(np.float32(dataset.load_image(val_input_names[ind])[:cfg.height, :cfg.width]), axis=0) / 255.0
                gt = dataset.load_image(val_output_names[ind])[:cfg.height, :cfg.width]
                gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

                # st = time.time()

                output_image = sess.run(network, feed_dict={net_input: input_image})

                output_image = np.array(output_image[0, :, :, :])
                output_image = helpers.reverse_one_hot(output_image)
                out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

                accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image,
                                                                                             label=gt,
                                                                                             num_classes=num_classes)

                file_name = utils.filepath_to_name(val_input_names[ind])
                target.write("%s, %f, %f, %f, %f, %f" % (file_name, accuracy, prec, rec, f1, iou))
                for item in class_accuracies:
                    target.write(", %f" % (item))
                target.write("\n")

                scores_list.append(accuracy)
                class_scores_list.append(class_accuracies)
                precision_list.append(prec)
                recall_list.append(rec)
                f1_list.append(f1)
                iou_list.append(iou)

                gt = helpers.colour_code_segmentation(gt, label_values)

                file_name = os.path.basename(val_input_names[ind])
                file_name = os.path.splitext(file_name)[0]
                cv2.imwrite(cfg.base_dir + "%s/%s/%04d/%s_pred.png" % ("checkpoints", cfg.base_dir, epoch, file_name),
                            cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
                cv2.imwrite(cfg.base_dir + "%s/%s/%04d/%s_gt.png" % ("checkpoints", cfg.base_dir, epoch, file_name),
                            cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

            target.close()

            avg_score = np.mean(scores_list)
            class_avg_scores = np.mean(class_scores_list, axis=0)
            avg_scores_per_epoch.append(avg_score)
            avg_precision = np.mean(precision_list)
            avg_recall = np.mean(recall_list)
            avg_f1 = np.mean(f1_list)
            avg_iou = np.mean(iou_list)

            print("\nAverage validation accuracy for epoch # %04d = %f" % (epoch, avg_score))
            print("Average per class validation accuracies for epoch # %04d:" % (epoch))
            for index, item in enumerate(class_avg_scores):
                print("%s = %f" % (class_names_list[index], item))
            print("Validation precision = ", avg_precision)
            print("Validation recall = ", avg_recall)
            print("Validation F1 score = ", avg_f1)
            print("Validation IoU score = ", avg_iou)

        epoch_time = time.time() - epoch_st
        remain_time = epoch_time * (cfg.num_epochs - 1 - epoch)
        m, s = divmod(remain_time, 60)
        h, m = divmod(m, 60)
        if s != 0:
            train_time = "Remaining training time = %d hours %d minutes %d seconds\n" % (h, m, s)
        else:
            train_time = "Remaining training time : Training completed.\n"
        utils.LOG(train_time)
        scores_list = []

    utils.drawLine(range(cfg.num_epochs), avg_scores_per_epoch, cfg.base_dir + 'checkpoints/' + cfg.model + '/accuracy_vs_epochs.png',
                   title='Average validation accuracy vs epochs', xlabel='Epoch', ylabel='Avg. val. accuracy')
    utils.drawLine(range(cfg.num_epochs), avg_loss_per_epoch, cfg.base_dir + 'checkpoints/' + cfg.model + '/loss_vs_epochs.png',
                   title='Average loss vs epochs', xlabel='Epoch', ylabel='Current loss')

def val():
    # Create directories if needed
    if not os.path.isdir(cfg.base_dir + "%s/%s" % ("result", "Val")):
        os.makedirs(cfg.base_dir + "%s/%s" % ("result", "Val"))

    target = open(cfg.base_dir + "%s/%s/val_scores.csv" % ("result", "Val"), 'w')
    target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))
    scores_list = []
    class_scores_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    iou_list = []
    run_times_list = []

    # Run testing on ALL test images
    for ind in range(len(val_input_names)):
        sys.stdout.write("\rRunning test image %d / %d" % (ind + 1, len(val_input_names)))
        sys.stdout.flush()

        input_image = np.expand_dims(np.float32(dataset.load_image(val_input_names[ind])[:cfg.height, :cfg.width]), axis=0) / 255.0
        gt = dataset.load_image(val_output_names[ind])[:cfg.height, :cfg.width]
        gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

        st = time.time()
        output_image = sess.run(network, feed_dict={net_input: input_image})

        run_times_list.append(time.time() - st)

        output_image = np.array(output_image[0, :, :, :])
        output_image = helpers.reverse_one_hot(output_image)
        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

        accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt,
                                                                                     num_classes=num_classes)

        file_name = utils.filepath_to_name(val_input_names[ind])
        target.write("%s, %f, %f, %f, %f, %f" % (file_name, accuracy, prec, rec, f1, iou))
        for item in class_accuracies:
            target.write(", %f" % (item))
        target.write("\n")

        scores_list.append(accuracy)
        class_scores_list.append(class_accuracies)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        iou_list.append(iou)

        gt = helpers.colour_code_segmentation(gt, label_values)

        cv2.imwrite(cfg.base_dir + "%s/%s/%s_pred.png" % ("result", "Val", file_name), cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
        cv2.imwrite(cfg.base_dir + "%s/%s/%s_gt.png" % ("result", "Val", file_name), cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

    target.close()

    avg_score = np.mean(scores_list)
    class_avg_scores = np.mean(class_scores_list, axis=0)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_iou = np.mean(iou_list)
    avg_time = np.mean(run_times_list)
    print("Average test accuracy = ", avg_score)
    print("Average per class test accuracies = \n")
    for index, item in enumerate(class_avg_scores):
        print("%s = %f" % (class_names_list[index], item))
    print("Average precision = ", avg_precision)
    print("Average recall = ", avg_recall)
    print("Average F1 score = ", avg_f1)
    print("Average mean IoU score = ", avg_iou)
    print("Average run time = ", avg_time)

def predict():

    # Equivalent to shuffling
    for test in test_input_names:

        # to get the right aspect ratio of the output
        loaded_image = dataset.load_image(test)
        height, width, channels = loaded_image.shape
        resize_height = int(height / (width / cfg.width))

        resized_image = cv2.resize(loaded_image, (cfg.width, resize_height))
        input_image = np.expand_dims(np.float32(resized_image[:cfg.height, :cfg.width]), axis=0) / 255.0

        st = time.time()
        output_image = sess.run(network, feed_dict={net_input: input_image})

        run_time = time.time() - st

        output_image = np.array(output_image[0, :, :, :])
        output_image = helpers.reverse_one_hot(output_image)

        # this needs to get generalized
        # class_names_list, label_values = helpers.get_label_info(os.path.join(cfg.data_dir, "class_dict.csv"))

        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
        out_vis_image = cv2.resize(out_vis_image, (height, width))
        out_vis_image[out_vis_image >= cfg.threshold*255] = 255
        out_vis_image[out_vis_image < cfg.threshold*255] = 0

        save_img = cv2.cvtColor(np.uint8(loaded_image), cv2.COLOR_RGB2BGR)
        transparent_image = np.append(np.array(save_img)[:, :, 0:3], out_vis_image[:, :, None], axis=-1)
        transparent_image = Image.fromarray(transparent_image)

        file_name = utils.filepath_to_name(test)
        cv2.imwrite(cfg.base_dir + "%s/%s/%s_pred.png" % ("result", "Test", file_name), transparent_image)
    print("Finished!")

if __name__ == '__mian__':
    # Load the data
    print("Loading the data ...")
    class_names_list, label_values = helpers.get_label_info(os.path.join(cfg.data_dir, 'class_dict.csv'))
    class_names_string = ""
    for class_name in class_names_list:
        if not class_name == class_names_list[-1]:
            class_names_string = class_names_string + class_name + ", "
        else:
            class_names_string = class_names_string + class_name
    num_classes = len(label_values)
    train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = dataset.prepare_data()


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Compute your softmax cross entropy loss
    print("Preparing the model ...")
    net_input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    net_output = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])

    network, init_fn = buildNetwork(cfg.model, net_input, num_classes)

    saver = tf.train.Saver(max_to_keep=cfg.num_keep)

    # Load a previous checkpoint if desired
    model_checkpoint_name = cfg.base_dir + "checkpoints/" + cfg.model + "/" + "model.ckpt"
    if cfg.continue_training or not cfg.mode == 'train':
        print('Loaded latest model checkpoint')
        saver.restore(sess, model_checkpoint_name)

    train()