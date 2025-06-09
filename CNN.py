import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Загрузка данных из нескольких CSV-файлов
def load_data_from_csvs(csv_files):
    data_frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Добавляем путь к директории для каждого CSV
        df['image_filename'] = df['image_filename'].apply(lambda x: os.path.join(os.path.dirname(csv_file), x))
        df['mask_filename'] = df['mask_filename'].apply(lambda x: os.path.join(os.path.dirname(csv_file), x) if pd.notnull(x) else None)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

# Загрузка изображений и масок
def load_images_and_masks(data):
    images = []
    masks = []
    labels = []
    for i in range(len(data)):
        image_path = data.iloc[i]['image_filename']
        mask_path = data.iloc[i]['mask_filename']
        label = data.iloc[i]['label']
        print(image_path)
        print(label)
        
        # Загрузка и предобработка изображений
        image = load_img(image_path, target_size=(360, 561))
        image = img_to_array(image) / 255.0
        images.append(image)
        
        # Загрузка и предобработка масок
        if mask_path and os.path.exists(mask_path):
            mask = load_img(mask_path, color_mode='grayscale', target_size=(360, 561))
            mask = img_to_array(mask) / 255.0
            masks.append(mask)
        else:
            masks.append(np.zeros((360, 561, 1)))  # Если ее нет, то создаем пустую маску
        
        # Добавление метки
        labels.append(label)
    
    return np.array(images), np.array(masks), np.array(labels)

# Загрузка данных из нескольких CSV-файлов
csv_files = [
    'Путь к файлам разметки', 
]  # Указать пути к CSV-файлам
data = load_data_from_csvs(csv_files)

images, masks, labels = load_images_and_masks(data)

# Разделение на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Подсчет весов классов для учета несбалансированности
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(y_train),
                                                  y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print(class_weights)
print(class_weights_dict)

unique, counts = np.unique(y_train, return_counts=True)
print(f"Training classes distribution: {dict(zip(unique, counts))}")

unique, counts = np.unique(y_test, return_counts=True)
print(f"Test classes distribution: {dict(zip(unique, counts))}")




# Настройка генератора данных с аугментацией
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.005,
    height_shift_range=0.005,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    #brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Генерация аугментированных изображений для примера
sample_image = x_train[np.random.randint(0, len(x_train))]
sample_image = np.expand_dims(sample_image, axis=0)

iterator = datagen.flow(sample_image, batch_size=1)

# Отображение оригинального и аугментированных изображений
fig, ax = plt.subplots(1, 5, figsize=(20, 5))

ax[0].imshow(sample_image[0])
ax[0].set_title('Original Image')
ax[0].axis('off')

for i in range(1, 5):
    batch = iterator.next()
    ax[i].imshow(batch[0])
    ax[i].set_title(f'Augmented Image {i}')
    ax[i].axis('off')

plt.show()

train_generator = datagen.flow(x_train, y_train, batch_size=70)

x_batch, y_batch = next(train_generator)
print(f"Batch labels: {y_batch}")

# Настройка генератора для проверки
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow(x_test, y_test, batch_size=70)

# Простая модель CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(360, 561, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

# Настройка TensorBoard

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

tensorboard_callback = TensorBoard(log_dir="path/to/logs", histogram_freq=1)

# Обучение модели с учетом весов классов
history = model.fit(train_generator, 
                    epochs=9, 
                    validation_data=test_generator, 
                    class_weight=class_weights_dict, 
                    steps_per_epoch=len(x_train) // 70, 
                    validation_steps=len(x_test) // 70,
                    callbacks=[tensorboard_callback, early_stopping_callback, checkpoint_callback])

predictions = model.predict(test_generator)
pred_labels = (predictions > 0.5).astype(int)
unique_preds, counts_preds = np.unique(pred_labels, return_counts=True)
print(f"Predictions distribution: {dict(zip(unique_preds, counts_preds))}")

for i in range(10):
    img = x_test[i]
    label = y_test[i]
    img_input = np.expand_dims(img, axis=0)
    prediction = model.predict(img_input)
    print(f"True label: {label}, Prediction: {prediction[0][0]}")
    plt.imshow(img)
    plt.title(f"True: {label}, Pred: {prediction[0][0]:.2f}")
    plt.axis('off')
    plt.show()


# Оценка модели
test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')
print(f'Test precision: {test_precision}')
print(f'Test recall: {test_recall}')
print(f'Test AUC: {test_auc}')

model.save('my_model_post.h5')

# Отображение результатов обучения
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8,8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Точность на обучении')
plt.plot(epochs_range, val_acc, label='Точность на валидации')
plt.legend(loc='lower right')
plt.title('Точность на обучающих и валидационных данных')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Потери на обучении')
plt.plot(epochs_range, val_loss, label='Потери на валидации')
plt.legend(loc='upper right')
plt.title('Потери на обучающих и валидационных данных')
plt.savefig('./training_results.png')
plt.show()
