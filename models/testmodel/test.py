from tensorflow import (sequence, )

def text_dnn(input_shape): #text query converted to vectors using word2vec 
    model = Sequential()
    model.add(Flatten(input_shape=input_shape)) 
    model.add(Dense(1024, activation='relu', name='text_combination'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', name='text_feature_vector'))
    return model

def image_cnn(input_shape,number_of_examples):
    images_inputs = [Input(image_input_shape) for i in range(5+1)]
    vgg_16= VGG16(weights='imagenet' ,input_shape=(224, 224, 3))
    vgg_16= Model(inputs=vgg_16.input, outputs=vgg_16.get_layer('fc2').output)
    model = Sequential()
    model.add(vgg_16)
    image_layer_combination = [model(image_input) for image_input in images_inputs]
    combination_layer  = concatenate(image_layer_combination)
    fc_1 = Dense(2048,activation='relu', name='image_combination')(combination_layer)
    image_model  = Model(inputs=images_inputs,outputs=fc_1)
    return image_model

def mergeCnnModel(cnnModel, cnnModel2):
    merged = concatenate([cnnModel.layers[-1].output, cnnModel2.layers[-1].output])
    dense1 = Dense(1024, activation='relu', name='image_text_combination')(merged)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(128, activation='relu', name='feature_vector')(drop1)
    outputs = Dense(1, activation='sigmoid')(dense2)
    model = Model(inputs=cnnModel.inputs+cnnModel2.inputs, outputs=outputs)
    return model


def get_complex_model(image_input_shape, number_of_examples,text_input_shape):

    #text branch
    text_model = text_dnn(text_input_shape)
    #images branch
    image_model=image_cnn(image_input_shape, number_of_examples)
    #merge branches
    new_model= mergeCnnModel(image_model,text_model )
    return new_model

combined_model  = get_complex_model(image_input_shape,number_of_examples,text_input_shape)