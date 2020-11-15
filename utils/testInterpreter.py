# import interpretdl as it
import InterpretDL.interpretdl as it
from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid
import numpy as np

from models.resnet18.model_with_code.model import x2paddle_net

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
    exe.run(fluid. default_startup_program())

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
        if labels is None:
            _, _, out = self.predict_fn(
                data, np.zeros(
                    (bsz, 1), dtype='int64'))
            labels = np.argmax(out, axis=1)
        labels = np.array(labels).reshape((bsz, 1))

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
        for i in range(bsz):
            visualize_heatmap(heatmap[i], imgs[i], visual, save_path[i])

        return heatmap, labels

sg = it.GradCAMInterpreter(paddle_model, model_path, use_cuda=False,
                           model_input_shape=[3, 224, 224])
gradients, labels, output = sg.interpret(img_path, visual=True, target_layer_name=last_layer_name, save_path='sg_test.jpg')

print(labels, output)
