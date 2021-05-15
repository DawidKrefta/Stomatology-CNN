import numpy as np
import keras
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

train_path = 'C:/Users/dawid/OneDrive/Pulpit/training'
test_path = 'C:/Users/dawid/OneDrive/Pulpit/test'
valid_path = 'C:/Users/dawid/OneDrive/Pulpit/validation'


def prepare_image(file):
    img_path = 'MobileNet-inference-images/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


train_batches = ImageDataGenerator(
    preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=train_path,
                                                                                              target_size=(224, 224),
                                                                                              batch_size=20)
valid_batches = ImageDataGenerator(
    preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=valid_path,
                                                                                              target_size=(224, 224),
                                                                                              batch_size=10)
test_batches = ImageDataGenerator(
    preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=test_path,
                                                                                              target_size=(224, 224),
                                                                                              batch_size=10,
                                                                                              shuffle=False)

model = keras.applications.resnet.ResNet50()
classes = list(iter(train_batches.class_indices))
model.layers.pop()
model.layers.pop()
for layer in model.layers:
    layer.trainable = False
last = model.layers[-1].output
x = Dense(classes, activation="softmax")(last)
finetuned_model = Model(model.input, x)
# for layer in model.layers[:-5]:
#     layer.trainable = False
finetuned_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = finetuned_model.fit_generator(generator=train_batches, steps_per_epoch=110,
                                        validation_data=valid_batches, validation_steps=60, epochs=3, verbose=2)

test_labels = test_batches.classes
predictions = finetuned_model.predict_generator(test_batches, steps=10, verbose=0)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))
cm_plot_labels = ['with', 'without']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
# plot dla accuracy
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
# plot dla loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
