from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers
import os
import numpy as np
import io
import skimage.io
from PIL import Image

NDFloat32Array = np.ndarray

current_dir = os.path.dirname(os.path.abspath(__file__))

# nsfw 실사 검출 모델 : https://github.com/bhky/opennsfw2
NSFW_WEIGHT_FROM_PHOTO = 'open_nsfw_weights.h5'
# nsfw 실사 + 애니메이션 검출 모델 : https://www.notion.so/snaps-corp/NSFW-Model-64912e8ec9534225aa08f7cf555b4fd7?pvs=4
NSFW_WEIGHT_FROM_PHOTO_WITH_ANIMATION = 'open_nsfw_weights_v_1.0.0.h5'


class NsfwModel(object):

    def __init__(self, weights_path='/var/task/'+NSFW_WEIGHT_FROM_PHOTO, threshold=0.9):
        self._model = self.get_model(weights_path)
        self._threshold = threshold

    def _batch_norm(self, name: str) -> layers.BatchNormalization:
        return layers.BatchNormalization(name=name, epsilon=1e-05)

    def _conv_block(self, stage: int, block: int, inputs: tf.Tensor, nums_filters: Tuple[int, int, int],
                    kernel_size: int = 3, stride: int = 2, ) -> tf.Tensor:
        num_filters_1, num_filters_2, num_filters_3 = nums_filters

        conv_name_base = f"conv_stage{stage}_block{block}_branch"
        bn_name_base = f"bn_stage{stage}_block{block}_branch"
        shortcut_name_post = f"_stage{stage}_block{block}_proj_shortcut"
        final_activation_name = f"activation_stage{stage}_block{block}"
        activation_name_base = f"{final_activation_name}_branch"

        shortcut = layers.Conv2D(
            name=f"conv{shortcut_name_post}",
            filters=num_filters_3,
            kernel_size=1,
            strides=stride,
            padding="same"
        )(inputs)

        shortcut = self._batch_norm(f"bn{shortcut_name_post}")(shortcut)

        x = layers.Conv2D(
            name=f"{conv_name_base}2a",
            filters=num_filters_1,
            kernel_size=1,
            strides=stride,
            padding="same"
        )(inputs)
        x = self._batch_norm(f"{bn_name_base}2a")(x)
        x = layers.Activation("relu", name=f"{activation_name_base}2a")(x)

        x = layers.Conv2D(
            name=f"{conv_name_base}2b",
            filters=num_filters_2,
            kernel_size=kernel_size,
            strides=1,
            padding="same"
        )(x)
        x = self._batch_norm(f"{bn_name_base}2b")(x)
        x = layers.Activation("relu", name=f"{activation_name_base}2b")(x)

        x = layers.Conv2D(
            name=f"{conv_name_base}2c",
            filters=num_filters_3,
            kernel_size=1,
            strides=1,
            padding="same"
        )(x)
        x = self._batch_norm(f"{bn_name_base}2c")(x)

        x = layers.Add()([x, shortcut])

        return layers.Activation("relu", name=final_activation_name)(x)

    def _identity_block(self, stage: int, block: int, inputs: tf.Tensor, nums_filters: Tuple[int, int, int],
                        kernel_size: int) -> tf.Tensor:
        num_filters_1, num_filters_2, num_filters_3 = nums_filters

        conv_name_base = f"conv_stage{stage}_block{block}_branch"
        bn_name_base = f"bn_stage{stage}_block{block}_branch"
        final_activation_name = f"activation_stage{stage}_block{block}"
        activation_name_base = f"{final_activation_name}_branch"

        x = layers.Conv2D(
            name=f"{conv_name_base}2a",
            filters=num_filters_1,
            kernel_size=1,
            strides=1,
            padding="same"
        )(inputs)
        x = self._batch_norm(f"{bn_name_base}2a")(x)
        x = layers.Activation("relu", name=f"{activation_name_base}2a")(x)

        x = layers.Conv2D(
            name=f"{conv_name_base}2b",
            filters=num_filters_2,
            kernel_size=kernel_size,
            strides=1,
            padding="same"
        )(x)
        x = self._batch_norm(f"{bn_name_base}2b")(x)
        x = layers.Activation("relu", name=f"{activation_name_base}2b")(x)

        x = layers.Conv2D(
            name=f"{conv_name_base}2c",
            filters=num_filters_3,
            kernel_size=1,
            strides=1,
            padding="same"
        )(x)
        x = self._batch_norm(f"{bn_name_base}2c")(x)

        x = layers.Add()([x, inputs])

        return layers.Activation("relu", name=final_activation_name)(x)

    def _preprocess_image(self, pil_image):

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        pil_image_resized = pil_image.resize((256, 256), resample=Image.BILINEAR)

        fh_im = io.BytesIO()
        pil_image_resized.save(fh_im, format="JPEG")
        fh_im.seek(0)

        image: NDFloat32Array = skimage.io.imread(
            fh_im, as_gray=False
        ).astype(np.float32)

        height, width, _ = image.shape
        h, w = (224, 224)

        h_off = max((height - h) // 2, 0)
        w_off = max((width - w) // 2, 0)
        image = image[h_off:h_off + h, w_off:w_off + w, :]

        # RGB to BGR
        image = image[:, :, ::-1]

        # Subtract the training dataset mean value of each channel.
        vgg_mean = [104, 117, 123]
        image = image - np.array(vgg_mean, dtype=np.float32)

        return image

    def get_model(self, weights_path, input_shape=(224, 224, 3)):

        image_input = layers.Input(shape=input_shape, name="input")
        x = image_input

        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        x = layers.Conv2D(name="conv_1", filters=64, kernel_size=7, strides=2,
                          padding="valid")(x)

        x = self._batch_norm("bn_1")(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

        x = self._conv_block(stage=0, block=0, inputs=x,
                             nums_filters=(32, 32, 128),
                             kernel_size=3, stride=1)

        x = self._identity_block(stage=0, block=1, inputs=x,
                                 nums_filters=(32, 32, 128), kernel_size=3)
        x = self._identity_block(stage=0, block=2, inputs=x,
                                 nums_filters=(32, 32, 128), kernel_size=3)

        x = self._conv_block(stage=1, block=0, inputs=x,
                             nums_filters=(64, 64, 256),
                             kernel_size=3, stride=2)
        x = self._identity_block(stage=1, block=1, inputs=x,
                                 nums_filters=(64, 64, 256), kernel_size=3)
        x = self._identity_block(stage=1, block=2, inputs=x,
                                 nums_filters=(64, 64, 256), kernel_size=3)
        x = self._identity_block(stage=1, block=3, inputs=x,
                                 nums_filters=(64, 64, 256), kernel_size=3)

        x = self._conv_block(stage=2, block=0, inputs=x,
                             nums_filters=(128, 128, 512),
                             kernel_size=3, stride=2)
        x = self._identity_block(stage=2, block=1, inputs=x,
                                 nums_filters=(128, 128, 512), kernel_size=3)
        x = self._identity_block(stage=2, block=2, inputs=x,
                                 nums_filters=(128, 128, 512), kernel_size=3)
        x = self._identity_block(stage=2, block=3, inputs=x,
                                 nums_filters=(128, 128, 512), kernel_size=3)
        x = self._identity_block(stage=2, block=4, inputs=x,
                                 nums_filters=(128, 128, 512), kernel_size=3)
        x = self._identity_block(stage=2, block=5, inputs=x,
                                 nums_filters=(128, 128, 512), kernel_size=3)

        x = self._conv_block(stage=3, block=0, inputs=x,
                             nums_filters=(256, 256, 1024), kernel_size=3,
                             stride=2)
        x = self._identity_block(stage=3, block=1, inputs=x,
                                 nums_filters=(256, 256, 1024),
                                 kernel_size=3)
        x = self._identity_block(stage=3, block=2, inputs=x,
                                 nums_filters=(256, 256, 1024),
                                 kernel_size=3)

        x = layers.GlobalAveragePooling2D()(x)

        logits = layers.Dense(name="fc_nsfw", units=2)(x)
        output = layers.Activation("softmax", name="predictions")(logits)

        model = tf.keras.Model(image_input, output)

        model.load_weights(weights_path)

        return model

    def inference(self, pil_image):

        result = {'type': -1, 'confidence': -1}
        image = self._preprocess_image(pil_image)
        nsfw_result = self._model.predict(np.expand_dims(image, 0), batch_size=1, verbose=0)[0][1]

        if nsfw_result > self._threshold:
            is_porn = 1
        else:
            is_porn = 0

        result['type'] = int(is_porn)
        result['confidence'] = float(nsfw_result)

        return result