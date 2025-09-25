# train.py — EfficientNetB0, preprocess đúng, không mixup

import os, json, math
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input

DATA_DIR    = "data"
IMG_SIZE    = (256, 256)
BATCH       = 32
EPOCHS_HEAD = 8
EPOCHS_FT   = 20
BASE_LR     = 3e-4
FT_LR       = 1e-4
LABEL_SMOOTH= 0.05
OUTPUT_H5   = "model.h5"
LABELS_JSON = "labels.json"
SEED        = 1337
AUTOTUNE    = tf.data.AUTOTUNE

# ép float32 (CPU)
keras.mixed_precision.set_global_policy("float32")

train_dir = Path(DATA_DIR)/"train"
val_dir   = Path(DATA_DIR)/"val"
test_dir  = Path(DATA_DIR)/"test"

classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
assert len(classes) >= 2
with open(LABELS_JSON,"w",encoding="utf-8") as f: json.dump(classes,f,ensure_ascii=False,indent=2)

def count_images(p: Path):
    exts={".jpg",".jpeg",".png",".bmp",".webp"}
    return sum(1 for fp in p.rglob("*") if fp.suffix.lower() in exts)
cls_counts={c: count_images(train_dir/c) for c in classes}
total=sum(cls_counts.values())
class_weight={i: total/(len(classes)*max(1,cls_counts[c])) for i,c in enumerate(classes)}
print("Class counts:", cls_counts)

def make_ds(root, shuffle=True):
    return image_dataset_from_directory(
        root, image_size=IMG_SIZE, batch_size=BATCH, seed=SEED,
        labels="inferred", label_mode="categorical", shuffle=shuffle)

train_ds=make_ds(train_dir,True)
val_ds  =make_ds(val_dir,False)
test_ds =make_ds(test_dir,False) if test_dir.exists() else None

augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.12),
    layers.RandomTranslation(0.06,0.06),
    layers.RandomContrast(0.15),
], name="augment")

def prep(x,y):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)      # *** dùng đúng preprocess của EfficientNet
    return x,y

train_ds=(train_ds.map(prep, num_parallel_calls=AUTOTUNE)
                  .map(lambda x,y:(augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
                  .cache().shuffle(2048, seed=SEED).prefetch(AUTOTUNE))
val_ds  =(val_ds.map(prep, num_parallel_calls=AUTOTUNE)
                .cache().prefetch(AUTOTUNE))
if test_ds: test_ds=(test_ds.map(prep, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE))

inputs=layers.Input(shape=(*IMG_SIZE,3))
base=EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")
base.trainable=False

x=base.output
x=layers.GlobalAveragePooling2D()(x)
x=layers.Dropout(0.35)(x)
x=layers.Dense(256, activation="relu")(x)
x=layers.Dropout(0.25)(x)
outputs=layers.Dense(len(classes), activation="softmax")(x)
model=keras.Model(inputs, outputs)

steps_per_epoch=max(1, math.ceil(total/BATCH))
cos_head=keras.optimizers.schedules.CosineDecay(BASE_LR, steps_per_epoch*EPOCHS_HEAD)
cos_ft  =keras.optimizers.schedules.CosineDecay(FT_LR,   steps_per_epoch*EPOCHS_FT)

loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH)
metrics=[keras.metrics.CategoricalAccuracy(name="acc")]

cbs=[
    keras.callbacks.ModelCheckpoint(OUTPUT_H5, monitor="val_acc", mode="max", save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_acc", mode="max"),
    keras.callbacks.CSVLogger("training_log.csv", append=False),
]

print("\n>>> Phase 1: train head")
model.compile(optimizer=keras.optimizers.Adam(cos_head), loss=loss, metrics=metrics)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
          class_weight=class_weight, callbacks=cbs, verbose=1)

print("\n>>> Phase 2: fine-tune")
for layer in base.layers[-100:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable=True

model.compile(optimizer=keras.optimizers.Adam(cos_ft), loss=loss, metrics=metrics)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT,
          class_weight=class_weight, callbacks=cbs, verbose=1)

if test_ds:
    print("\nTest evaluation:", model.evaluate(test_ds, verbose=0))

model.save(OUTPUT_H5)
print(f"\nSaved: {OUTPUT_H5} & {LABELS_JSON}")
