##edited on 300519
import os
from keras import backend as K
import cv2

from keras.layers import Input
from keras.models import Model
import roi_helpers

import resnet as nn

import io
import base64
import scipy


import numpy as np

class MyPredictor(object):
        """Interface for constructing custom predictors."""
    def __init__(self, config, model_rpn, model_classifier):
        self._config =config#this config holds the updated config dict edited in from_path method## {'verbose': True, 'network': 'resnet50', 'use_horizontal_flips': False, 'use_vertical_flips': False, 'rot_90': False, 'anchor_box_scales': [128, 256, 512], 'anchor_box_ratios': [[1, 1], [0.7071067811865475, 1.414213562373095], [1.414213562373095, 0.7071067811865475]], 'im_size': 600, 'img_channel_mean': [103.939, 116.779, 123.68], 'img_scaling_factor': 1.0, 'num_rois': 32, 'rpn_stride': 16, 'balanced_classes': False, 'std_scaling': 4.0, 'classifier_regr_std': [8.0, 8.0, 4.0, 4.0], 'rpn_min_overlap': 0.3, 'rpn_max_overlap': 0.7, 'classifier_min_overlap': 0.1, 'classifier_max_overlap': 0.5, 'class_mapping': {'cake': 0, 'donuts': 1, 'dosa': 2, 'bg': 3}, 'model_path': './model_frcnn.hdf5', 'base_net_weights': 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'} #240519# manually entering dictionary values here config #loads the config file(as dictionary) generated after training
        self._model_rpn= model_rpn
        self._model_classifier= model_classifier

    def format_img_size(self, img):
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
    def format_img_channels(self, img):
        """ formats the image channels based on config """
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= self._config['img_channel_mean'][0]
        img[:, :, 1] -= self._config['img_channel_mean'][1]
        img[:, :, 2] -= self._config['img_channel_mean'][2]
        img /= self._config['img_scaling_factor']
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    
    def format_img(self, img):
        """ formats an image for model prediction based on config """
        img, ratio = self.format_img_size(img)
        img = self.format_img_channels(img)
        return img, ratio
    
    def get_real_coordinates(self, ratio, x1, y1, x2, y2):
        ratio= ratio
        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))
        return (real_x1, real_y1, real_x2 ,real_y2)
    
    def preprocess(self, inputs):
        X, ratio = self.format_img(inputs)
        
        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))
            
        [Y1, Y2, F] = self._model_rpn.predict(X)
        R = roi_helpers.rpn_to_roi(Y1, Y2, self._config, K.image_dim_ordering(), overlap_thresh=0.7)

        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]
        
        bboxes = {}
        probs = {}
        bbox_threshold = 0.8
        
        class_mapping= self._config['class_mapping']
        
        for jk in range(R.shape[0]//self._config['num_rois'] + 1):
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

            [P_cls, P_regr] = self._model_classifier.predict([F, ROIs])
            
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

        return [bboxes, probs, ratio]

    def postprocess(self, bounding_boxes, probabilities, ratio):
        all_dets=[]
        bboxes=bounding_boxes
        probs=probabilities
        ratio= ratio
        
        for key in bboxes:
            bbox = np.array(bboxes[key])
            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)

            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]
                coord_list= list(self.get_real_coordinates(ratio, x1, y1, x2, y2))#220519# addded self. to call class function
                all_dets.append((key,100*new_probs[jk],coord_list))

        return all_dets
        
    def predict(self, instances):
        
        inputs= base64.b64decode(instances['image_bytes']['b64'])# decodes the b64 string from json dictionary
        
        inputs= scipy.misc.imread(io.BytesIO(inputs))# converts unicode string to numpy array of image
        
        inputs= inputs[...,::-1]#170619 changes RGB(usual outcome of scipy) to BGR(recommended by cv2 locally)
        [bboxes, probs, ratio]= self.preprocess(inputs)
        
        results = self.postprocess(bboxes, probs, ratio)

        return results

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
        model_path= os.path.join(model_dir,'model_frcnn.hdf5')#220519# added full/correct path to model_weights in gcloud 
        num_features = 1024
        #config defined explicitly
        config ={'verbose': True, 'network': 'resnet50', 'use_horizontal_flips': False, 'use_vertical_flips': False, 'rot_90': False, 'anchor_box_scales': [128, 256, 512], 'anchor_box_ratios': [[1, 1], [0.7071067811865475, 1.414213562373095], [1.414213562373095, 0.7071067811865475]], 'im_size': 600, 'img_channel_mean': [103.939, 116.779, 123.68], 'img_scaling_factor': 1.0, 'num_rois': 32, 'rpn_stride': 16, 'balanced_classes': False, 'std_scaling': 4.0, 'classifier_regr_std': [8.0, 8.0, 4.0, 4.0], 'rpn_min_overlap': 0.3, 'rpn_max_overlap': 0.7, 'classifier_min_overlap': 0.1, 'classifier_max_overlap': 0.5, 'class_mapping': {'cake': 0, 'donuts': 1, 'dosa': 2, 'bg': 3}, 'model_path': './model_frcnn.hdf5', 'base_net_weights': 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'}
        
        class_mapping = config['class_mapping']
        if 'bg' not in class_mapping:
            class_mapping['bg'] = len(class_mapping)        
        class_mapping = {v: k for k, v in class_mapping.items()}        
        
        config['class_mapping']= class_mapping
        
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)
        
        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(config['num_rois'], 4))
        feature_map_input = Input(shape=input_shape_features)
        
        shared_layers = nn.nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(config['anchor_box_scales']) * len(config['anchor_box_ratios'])
        rpn_layers = nn.rpn(shared_layers, num_anchors)
        classifier = nn.classifier(feature_map_input, roi_input, config['num_rois'], nb_classes=len(class_mapping), trainable=True)
        model_rpn = Model(img_input, rpn_layers)
        
        model_classifier = Model([feature_map_input, roi_input], classifier)
        model_rpn.load_weights(model_path, by_name=True)
        
        model_classifier.load_weights(model_path, by_name=True)
        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')
        
        return cls(config, model_rpn, model_classifier)
