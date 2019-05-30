import os
import pickle

from keras import backend as K#140519
import cv2#140519

#import sys
from keras.layers import Input
from keras.models import Model

import roi_helpers#160519 #from keras_frcnn import roi_helpers
import numpy as np
import tensorflow as tf

class Predictor(object):
    """Interface for constructing custom predictors."""
    def __init__(self,model_path, config):
        self.path= model_path
        self._config = config#loads a dictionary, thus values are accessed as config['xyz'] instead config.xyz(Now replacing every config.xyz with config['xyz'])

    def format_img_size(self, img):#, C):#140519
        """ formats the image size based on config """
        img_min_side = float(self._config['im_size'])
        (height,width,_) = img.shape
        if width <= height:
            ratio = img_min_side/width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side/height
            new_width = int(ratio * width)
            new_height = int(img_min_side)

        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio

    def format_img_channels(self, img):#, C):#140519
        """ formats the image channels based on config """
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= self._config.img_channel_mean[0]
        img[:, :, 1] -= self._config.img_channel_mean[1]
        img[:, :, 2] -= self._config.img_channel_mean[2]
        img /= self._config.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def format_img(self, img):#, C): #140519
        """ formats an image for model prediction based on config """
        img, ratio = format_img_size(img)
        img = format_img_channels(img)
        return img, ratio
    
    def get_real_coordinates(ratio, x1, y1, x2, y2):#140519
        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))

        return (real_x1, real_y1, real_x2 ,real_y2)      



    def predict(self, instances):#, **kwargs):
        """Performs custom prediction.

        Instances are the decoded values from the request. They have already
        been deserialized from JSON.

        Args:
            instances: A list of prediction input instances.
            **kwargs: A dictionary of keyword args provided as additional
                fields on the predict request body.

        Returns:
            A list of outputs containing the prediction results. This list must
            be JSON serializable.
        """
        if self._config['network'] == 'resnet50':
            import resnet as nn
            num_features = 1024
        elif self._config['network'] == 'vgg':
            import vgg as nn
            num_features = 512
        #turn off any data augmentation at test time
        self._config['use_horizontal_flips'] = False
        self._config['use_vertical_flips'] = False
        self._config['rot_90'] = False

        class_mapping = self._config['class_mapping']

        if 'bg' not in class_mapping:
            class_mapping['bg'] = len(class_mapping)

        class_mapping = {v: k for k, v in class_mapping.items()}
        
        #print(class_mapping)
        class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
        self._config['num_rois'] = 32#140519

        if K.image_dim_ordering() == 'th':
            input_shape_img = (3, None, None)
            input_shape_features = (num_features, None, None)
        else:
            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, num_features)

        ####this all could be moved to classmethod below later
        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self._config['num_rois'], 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)
        # define the RPN, built on the base layers

        num_anchors = len(self._config['anchor_box_scales']) * len(self._config['anchor_box_ratios'])
        rpn_layers = nn.rpn(shared_layers, num_anchors)

        classifier = nn.classifier(feature_map_input, roi_input, self._config['num_rois'], nb_classes=len(class_mapping), trainable=True)

        model_rpn = Model(img_input, rpn_layers)

        model_classifier = Model([feature_map_input, roi_input], classifier)
    
        #150519 #explicit model_path could also be given to folder actually holding weights
        model_rpn.load_weights(self.path, by_name=True)#Now model_path is 'gs://{bucket_name}/model_frcnn.hdf5'
        model_classifier.load_weights(self.path, by_name=True)
    
        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')

        all_imgs = []
        classes = {}
        

        inputs = np.asarray(instances)#receives input in byte string format from image loaded from gcloud storage

        X, ratio = self.format_img(inputs)#220519#, C)#140519
        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        [Y1, Y2, F] = model_rpn.predict(X)
        R = roi_helpers.rpn_to_roi(Y1, Y2, self._config, K.image_dim_ordering(), overlap_thresh=0.7)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        bboxes = {}
        probs = {}
        bbox_threshold = 0.8

        for jk in range(R.shape[0]//C.num_rois + 1):
            ROIs = np.expand_dims(R[self._config['num_rois']*jk:self._config['num_rois']*(jk+1), :], axis=0)
            if ROIs.shape[1] == 0:
                break
            if jk == R.shape[0]//self._config['num_rois']:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],self._config['num_rois'],curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded
            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])
            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue
                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
                
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                (x, y, w, h) = ROIs[0, ii, :]
                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                    tx /= self._config['classifier_regr_std'][0]
                    ty /= self._config['classifier_regr_std'][1]
                    tw /= self._config['classifier_regr_std'][2]
                    th /= self._config['classifier_regr_std'][3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append([self._config['rpn_stride']*x, self._config['rpn_stride']*y, self._config['rpn_stride']*(x+w), self._config['rpn_stride']*(y+h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))
        all_dets = []

        for key in bboxes:
            bbox = np.array(bboxes[key])
            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)

            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]
                #(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                coord_list= list(self.get_real_coordinates(ratio, x1, y1, x2, y2))#220519#
                #print('norm coord:',real_x1, real_y1, real_x2, real_y2, '\n \n coord_list',coord_list)
                all_dets.append((key,100*new_probs[jk],coord_list))
        return all_dets
    
        #raise NotImplementedError()
    

    @classmethod
    def from_path(cls, model_dir):
        """Creates an instance of Predictor using the given path.

        Loading of the predictor should be done in this method.

        Args:
            model_dir: The local directory that contains the exported model
                file along with any additional files uploaded when creating the
                version resource.

        Returns:
            An instance implementing this Predictor class.
        """
        model_path= os.path.join(model_dir,'model_frcnn.hdf5')#adding full path to model in cloud storage
        config_path = os.path.join(model_dir, 'config.pickle')#config file is a dictionary holding key parameters
        with open(config_path, 'rb') as f_in:#140519# load config file from local path here
            config = pickle.load(f_in)
        #NOTE:data unpickled here to config is no more an instance of Class Config, instead confi.__dict__ dumped to a new pickle file; which now holds a mere dictionary. 

        return cls(model_path, config)
        #raise NotImplementedError()
