try:
    import keras
    from keras import backend as K
    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D 
    from keras.utils import to_categorical
    from keras.applications.vgg16 import preprocess_input
    from keras.preprocessing.image import ImageDataGenerator
    from keras.optimizers import Adam, SGD
    from keras.applications import VGG16
    from keras.applications.vgg16 import preprocess_input
    from keras.losses import binary_crossentropy
    from keras.callbacks import LearningRateScheduler
    from PIL import Image
except:
    raise ImportError("For this example you need to install keras.")

import numpy as np
import os
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)

class KerasWorker(Worker):
    def __init__(self, rootDir, batch_size=64, **kwargs):
        super().__init__(**kwargs)
        
        #global variables
        self.batch_size= batch_size                            #batch_size
        self.rootDir = rootDir                                 #root directory
        
        self.train_dir = os.path.join(self.rootDir , 'train')  #train directory
        self.val_dir = os.path.join(self.rootDir, 'val')     #validation directory
        self.test_dir = os.path.join(self.rootDir, 'test')   #test directory
        
        self.input_shape = (224, 224, 3)                       #input shape
        self.num_classes=2                                     #num output
        
        #Training Set data Augmentation
        self.data_processing = ImageDataGenerator(preprocessing_function=preprocess_input, 
                                             horizontal_flip=True,
                                             width_shift_range=0.2,
                                             height_shift_range=0.2,
                                             #validation_split=0.2,
                                             rotation_range=30,
                                             zoom_range=0.15,
                                             shear_range=0.15,
                                             fill_mode="nearest"
                                             )
        
        #Validation/Test Set Data (no) augmentation
        self.data_processing_val_test = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        '''
        #to be used only if categorical classification is desired
        #then output layer must have 2 neurons and softmax activation function
        
        self.y_train = to_categorical(self.y_train)
        self.y_val = to_categorical(self.y_val)
        self.y_test = to_categorical(self.y_test)
        '''    




        #Loading VGG16
        print('Loading VGG16...')
        self.base_net = VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False, pooling='avg')
        print('VGG16 Loaded')
        
        #Layers dictionary
        self.layers_dict = dict([(layer.name, layer) for layer in self.base_net.layers])

        for layer in self.base_net.layers:         
            layer.trainable = False    


    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Fine-tuning of VGG16 with HPO.
        Different trials on cut level, FC NAS and training hyperparameters
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        #Test Set data Generator    
        train_generator = self.data_processing.flow_from_directory(
                self.train_dir,
                target_size=(224, 224),
                color_mode="rgb",
                batch_size=self.batch_size,
                class_mode="binary",
                shuffle=True,
                seed=1,
                #subset='training'
                )

        #Validation Set Generator
        val_generator = self.data_processing_val_test.flow_from_directory(
                self.val_dir,
                target_size=(224, 224),
                color_mode="rgb",
                batch_size=self.batch_size,
                class_mode="binary",
                shuffle=True,
                seed=1,
                #subset='validation'
                )

        #Test Set data Generator
        test_generator = self.data_processing_val_test.flow_from_directory(
                self.test_dir,
                batch_size=self.batch_size,
                target_size=(224, 224),
                color_mode="rgb",
                class_mode="binary",
                shuffle=True,
                seed=1,
                #subset='test'
                )

        #1st Cut Level: all CONV layers freezed:
        #Training with HPO
        
        #Output del modello di base
        x = self.base_net.output
        #Nuovo livello fully-connected intermedio + ReLU
        x = Dense(config['num_fc_units_1'], activation='relu')(x)
        #Nuovo livello di Dvropout
        x = Dropout(config['dropout_rate_1'])(x)
        
        #Nuovo livello fully-connected intermedio + ReLU + Dropout
        if config['num_fc_layers']>1:
            x = Dense(config['num_fc_units_2'], activation='relu')(x)
            x = Dropout(config['dropout_rate_2'])(x)
        
        #Nuovo livello fully-connected finale + softmax
        pred = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=self.base_net.input, outputs=pred) 

        #freezing CONV layers
        for layer in model.layers:         
            layer.trainable = False 
        for layer in model.layers[20:]:         
            layer.trainable = True               
        
        if config['optimizer'] == 'Adam':
                optimizer_pre = Adam(1e-4)
                optimizer_post = Adam(config['lr'])
        else:
                optimizer_pre = SGD(1e-4, momentum=config['sgd_momentum'])
                optimizer_post = SGD(config['lr'], momentum=config['sgd_momentum'])

        #Compiling with very low learning rate for warmstarting
        model.compile(loss = binary_crossentropy,
                optimizer = optimizer_pre,
                metrics=['accuracy'])
        
        #pre-training
        model.fit(train_generator,
                epochs=10,
                verbose=0,
                validation_data=(val_generator))  

        warm_score = model.evaluate(val_generator, verbose=0)     

        #Unfreezing last convolutional block
        for layer in model.layers:         
            layer.trainable = False
        for layer in model.layers[int(config['cut_level']):]:         
            layer.trainable = True               

        train_generator.reset()
        val_generator.reset()
        test_generator.reset()
        '''
        #Re-compiling with updated lr             
        model.compile(loss = binary_crossentropy,
                optimizer = optimizer_post,
                metrics=['accuracy'])                 
        '''
        model.optimizer= optimizer_post

        #Fine-tuning training
        model.fit(train_generator,
                epochs=int(budget),
                verbose=0,
                validation_data=(val_generator))
        #scores
        train_score = model.evaluate(train_generator, verbose=0)
        val_score = model.evaluate(val_generator, verbose=0)
        test_score = model.evaluate(test_generator, verbose=0)

        return ({
                'loss': 1-val_score[1], # remember: HpBandSter always minimizes!
                'info': {       'test accuracy': test_score[1],
                                'train accuracy': train_score[1],
                                'validation accuracy': val_score[1],
                                'warm-starting accuracy': warm_score[1],
                                'number of parameters': model.count_params(),
                                }

        })

    @staticmethod
    def get_configspace():
            """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
            cs = CS.ConfigurationSpace()
            
            #adding optimizer, learning rate and momentum to CS
            optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])
            lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=1e-3, default_value='1e-4', log=True)
            sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)

            cs.add_hyperparameters([lr, optimizer, sgd_momentum])

            #adding NAS hyperparameters to CS
            #(number of fully-connected layers, dropout rates and number of units per layer)
            num_fc_layers = CSH.UniformIntegerHyperparameter('num_fc_layers', lower=1, upper=2, default_value=1, log=True)            
            dropout_rate_1 = CSH.UniformFloatHyperparameter('dropout_rate_1', lower=0.01, upper=0.5, default_value=0.2, log=True)
            dropout_rate_2 = CSH.UniformFloatHyperparameter('dropout_rate_2', lower=0.01, upper=0.5, default_value=0.2, log=True)
            num_fc_units_1 = CSH.UniformIntegerHyperparameter('num_fc_units_1', lower=8, upper=256, default_value=32, log=True)
            num_fc_units_2 = CSH.UniformIntegerHyperparameter('num_fc_units_2', lower=8, upper=256, default_value=32, log=True)

            cs.add_hyperparameters([num_fc_layers, dropout_rate_1, dropout_rate_2, num_fc_units_1, num_fc_units_2])

            #adding VGG16 cut level to CS
            cut_level = CSH.CategoricalHyperparameter('cut_level', [7,11,15,20])

            cs.add_hyperparameters([cut_level])
  
            #equality conditions
            cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
            cs.add_condition(cond)
            
            cond = CS.EqualsCondition(num_fc_units_2, num_fc_layers, 2)
            cs.add_condition(cond)
            
            cond = CS.EqualsCondition(dropout_rate_2, num_fc_layers, 2)
            cs.add_condition(cond)
                
            return cs

if __name__ == "__main__":
    worker = KerasWorker(run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, working_directory='.')
    print(res)
