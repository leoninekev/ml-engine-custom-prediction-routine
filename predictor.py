import os
import pickle



class Predictor(object):
    """Interface for constructing custom predictors."""
    def __init__(self,config):#170519 model, preprocessor):
        self._config = config#loads the config file instance generated after training

        

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
        if self._config.network == 'resnet50':
            import resnet as nn
            num_features = 1024
        elif self._config.network == 'vgg':
            import vgg as nn
            num_features = 512
        #turn off any data augmentation at test time
        self._config.use_horizontal_flips = False
        self._config.use_vertical_flips = False
        self._config.rot_90 = False
        class_mapping = self._config.class_mapping

        if 'bg' not in class_mapping:
            class_mapping['bg'] = len(class_mapping)

        class_mapping = {v: k for k, v in class_mapping.items()}
        
        #print(class_mapping)
        class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
        self._config.num_rois = 32#140519

        if K.image_dim_ordering() == 'th':
            input_shape_img = (3, None, None)
            input_shape_features = (num_features, None, None)
        else:
            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, num_features)
        ####this all could be moved to classmethod below
        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self._config.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)
        # define the RPN, built on the base layers

        num_anchors = len(self._config.anchor_box_scales) * len(self._config.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)

        classifier = nn.classifier(feature_map_input, roi_input, self._config.num_rois, nb_classes=len(class_mapping), trainable=True)

        model_rpn = Model(img_input, rpn_layers)
        model_classifier_only = Model([feature_map_input, roi_input], classifier)

        model_classifier = Model([feature_map_input, roi_input], classifier)
        #print('Loading weights from {}'.format(C.model_path))

        model_rpn.load_weights(self._config.model_path, by_name=True)
        #150519 #explicit model_path could also be given to folder actually holding weights
        model_classifier.load_weights(self._config.model_path, by_name=True)
    
        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')
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
        config_path = os.path.join(model_dir, 'config.pickle')#config file instance of class holding key parameters
        with open(config_path, 'rb') as f_in:#140519# load config file from local path here
            config = pickle.load(f_in)

        return cls(config)
        #raise NotImplementedError()
