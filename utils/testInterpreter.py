# import interpretdl as it
import os
from copy import copy

import cv2
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_dilation

import utils.InterpretDL.interpretdl as it
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid
import numpy as np

from utils.models.resnet18.model_with_code.model import x2paddle_net

from utils.InterpretDL.interpretdl.data_processor.readers import preprocess_inputs
from utils.InterpretDL.interpretdl.data_processor.visualizer import visualize_heatmap

last_layer_name = 'x2paddle_188.tmp_0'
model_path = 'models/resnet18/model_with_code'
test_result = 0
test_prob = 0
test_dic = {0: 'normal', 1: 'pun', 2: 'covid19'}


def paddle_model(data):
    inputs, outputs = x2paddle_net(input=data)
    ops = fluid.default_main_program().global_block().ops
    used_vars = list()
    for op in ops:
        used_vars += op.input_arg_names
    tmp = list()
    for input in inputs:
        if isinstance(input, list):
            for ipt in input:
                if ipt.name not in used_vars:
                    continue
                tmp.append(ipt)
        else:
            if input.name not in used_vars:
                continue
            tmp.append(input)
    inputs = tmp
    for i, out in enumerate(outputs):
        if isinstance(out, list):
            for out_part in out:
                outputs.append(out_part)
            del outputs[i]
    outputs = outputs[0]
    probs = fluid.layers.softmax(outputs, axis=-1)
    return probs


def get_result(data, param_dir="./"):
    import os
    inputs, outputs = x2paddle_net(input=data)
    ops = fluid.default_main_program().global_block().ops
    used_vars = list()
    for op in ops:
        used_vars += op.input_arg_names
    tmp = list()
    for input in inputs:
        if isinstance(input, list):
            for ipt in input:
                if ipt.name not in used_vars:
                    continue
                tmp.append(ipt)
        else:
            if input.name not in used_vars:
                continue
            tmp.append(input)
    inputs = tmp
    for i, out in enumerate(outputs):
        if isinstance(out, list):
            for out_part in out:
                outputs.append(out_part)
            del outputs[i]
    outputs = outputs[0]
    probs = fluid.layers.softmax(outputs, axis=-1)
    exe = fluid.Executor(fluid.CPUPlace())
    exe.run(fluid.default_startup_program())

    def if_exist(var):
        b = os.path.exists(os.path.join(param_dir, var.name))
        return b

    fluid.io.load_vars(
        exe, param_dir, fluid.default_main_program(), predicate=if_exist)
    exe.run(fluid.default_startup_program())


img_path = '/gui/utils/0a6a5956-58cf-4f17-9e39-7e0d17310f67.png'
img_path = '0a6a5956-58cf-4f17-9e39-7e0d17310f67.png'
img_path = '1-s2.0-S1684118220300608-main.pdf-002.jpg'
# img_path = 'utils/InterpretDL/tutorials/assets/catdog.png'
model_path = 'models/resnet18/model_with_code'


class MyGradCAMInterpreter(it.GradCAMInterpreter):
    def __init__(self,
                 paddle_model,
                 trained_model_path,
                 use_cuda=True,
                 model_input_shape=[3, 224, 224]) -> None:
        it.GradCAMInterpreter.__init__(self, paddle_model, trained_model_path, use_cuda, model_input_shape)

    def interpret(self,
                  inputs,
                  target_layer_name,
                  thresholds=0.5,
                  labels=None,
                  visual=True,
                  save_path=None):
        """
        Main function of the interpreter.

        Args:
            inputs (str or list of strs or numpy.ndarray): The input image filepath or a list of filepaths or numpy array of read images.
            target_layer_name (str): The target layer to calculate gradients.
            labels (list or tuple or numpy.ndarray, optional): The target labels to analyze. The number of labels should be equal to the number of images. If None, the most likely label for each image will be used. Default: None
            visual (bool, optional): Whether or not to visualize the processed image. Default: True
            save_path (str or list of strs or None, optional): The filepath(s) to save the processed image(s). If None, the image will not be saved. Default: None

        :return: interpretations/heatmap for each image
        :rtype: numpy.ndarray

        Example::

            import interpretdl as it
            def paddle_model(data):
                import paddle.fluid as fluid
                class_num = 1000
                model = ResNet50()
                logits = model.net(input=image_input, class_dim=class_num)
                probs = fluid.layers.softmax(logits, axis=-1)
                return probs
            gradcam = it.GradCAMInterpreter(paddle_model, "assets/ResNet50_pretrained",True)
            gradcam.interpret(
                    'assets/catdog.png',
                    'res5c.add.output.5.tmp_0',
                    label=None,
                    visual=True,
                    save_path='assets/gradcam_test.jpg')
        """

        imgs, data, save_path = preprocess_inputs(inputs, save_path,
                                                  self.model_input_shape)

        self.target_layer_name = target_layer_name

        if not self.paddle_prepared:
            self._paddle_prepare()

        bsz = len(data)
        ioutput = None
        if labels is None:
            _, _, out = self.predict_fn(
                data, np.zeros(
                    (bsz, 1), dtype='int64'))
            # labels = np.argmax(out, axis=1)
            # labels = np.array(labels).reshape((bsz, 1))
            # print(labels)
            ioutput = out
            print(out)
            data_t = copy(data)
            data = []
            labels = []
            for dt, ou in zip(data_t, out):
                for i in range(len(ou)):
                    if ou[i] > thresholds:
                        data.append(dt)
                        labels.append(i)
            # labels = np.argmax(out, axis=1)
        bsz = len(data)
        data = np.array(data)
        labels = np.array(labels).reshape((bsz, 1))
        print(labels)

        feature_map, gradients, _ = self.predict_fn(data, labels)

        f = np.array(feature_map)
        g = np.array(gradients)
        # print(f.shape, g.shape)
        mean_g = np.mean(g, (2, 3))
        heatmap = f.transpose([0, 2, 3, 1])
        dim_array = np.ones((1, heatmap.ndim), int).ravel()
        dim_array[heatmap.ndim - 1] = -1
        dim_array[0] = bsz
        heatmap = heatmap * mean_g.reshape(dim_array)

        heatmap = np.mean(heatmap, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap_max = np.max(heatmap, axis=tuple(np.arange(1, heatmap.ndim)))
        heatmap /= heatmap_max.reshape((bsz,) + (1,) * (heatmap.ndim - 1))

        try:
            os.makedirs(save_path[0])
        except:
            pass

        for i in range(bsz):
            visualize_heatmap(heatmap[i], imgs[0], visual, '{}/{}.jpg'.format(save_path[0], i))

        return heatmap, labels, ioutput, imgs[0]


def mix_heatmap(heatmap, org, label):
    org = np.array(org).astype('float32')
    org = cv2.cvtColor(org, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (org.shape[1], org.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    x = heatmap * 0.5 + org * 0.7
    x = np.clip(x, 0, 255)
    x = np.uint8(x)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    k, xx, yy, ww, hh = label
    color = (255, 0, 0)
    x = cv2.rectangle(x, (xx, yy), (ww, hh), color, 3)
    x = cv2.putText(x, k, (xx, yy-20), cv2.FONT_HERSHEY_COMPLEX, 3, color, 5)
    x = Image.fromarray(x)
    import matplotlib.pyplot as plt
    plt.imshow(x)
    plt.show()
    return x


def plot_bounding_box(labels, heatmaps, org, save_path):
    import scipy.ndimage.filters as filters
    img_width, img_height = 224, 224

    crop_del = 16
    rescale_factor = 1
    class_index = ['pneumonia', 'normal', 'COVID-19']
    avg_size = np.array([[411.8, 512.5, 276.5, 304.5], [411.8, 512.5, 276.5, 304.5],
                         [411.8, 512.5, 276.5, 304.5]])

    '''
    class_index = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                   'Pneumothorax']
    avg_size = np.array([[411.8, 512.5, 219.0, 139.1], [348.5, 392.3, 479.8, 381.1],
                         [396.5, 415.8, 221.6, 318.0], [394.5, 389.1, 294.0, 297.4],
                         [434.3, 366.7, 168.7, 189.8], [502.4, 458.7, 71.9, 70.4],
                         [378.7, 416.7, 276.5, 304.5], [369.3, 209.4, 198.9, 246.0]])
    '''
    bbox = []
    cnt = 0
    for label, heatmap in zip(labels, heatmaps):
        label = label[0]
        print(heatmap.shape)
        data = heatmap
        # data = heatmap.reshape(-1, img_width, img_height)

        # output avgerge
        prediction_sent = '%s %.1f %.1f %.1f %.1f' % (
            class_index[label], avg_size[label][0], avg_size[label][1], avg_size[label][2], avg_size[label][3])

        if np.isnan(data).any():
            continue

        w_k, h_k = (avg_size[label][2:4] * (256 / 1024)).astype(np.int)

        # Find local maxima
        neighborhood_size = 100
        threshold = .1

        data_max = filters.maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        for _ in range(5):
            maxima = binary_dilation(maxima)

        labeled, num_objects = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)
        xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects + 1)))
        print(xy)

        for pt in xy:
            upper = int(max(pt[0] - (h_k / 2), 0.))
            left = int(max(pt[1] - (w_k / 2), 0.))

            right = int(min(left + w_k, img_width))
            lower = int(min(upper + h_k, img_height))

            prediction_sent = '%s %.1f %.1f %.1f %.1f' % (class_index[label], (left + crop_del) * rescale_factor,
                                                          (upper + crop_del) * rescale_factor,
                                                          (right - left) * rescale_factor,
                                                          (lower - upper) * rescale_factor)
            bbox.append([class_index[label], (left + crop_del) * rescale_factor,
                         (upper + crop_del) * rescale_factor,
                         (right - left) * rescale_factor,
                         (lower - upper) * rescale_factor])
            print(bbox[-1])
            cnt += 1
            x = mix_heatmap(heatmap, org, bbox[-1])
            x.save('{}/{}_with_bbox.jpg'.format(save_path, cnt))


if __name__ == '__main__':
    sg = it.GradCAMInterpreter(paddle_model, model_path, use_cuda=False,
                               model_input_shape=[3, 224, 224])
    gradients, labels, output = sg.interpret(img_path, visual=True, target_layer_name=last_layer_name,
                                             save_path='sg_test.jpg')
    print(labels, output)
