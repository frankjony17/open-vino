from __future__ import print_function

import os
from pathlib import Path

import cv2
from fastapi import HTTPException

from openvino.inference_engine import IENetwork, IEPlugin
from src.util.api_util import ApiUtil


class FaceDetectionService:

    __prob_threshold = 0.5
    __cpu_extension = str(Path(__file__).resolve().parent.parent.parent.
                          joinpath("lib/libcpu_extension.so"))
    __static_path = str(Path(__file__).resolve().parent.parent.parent.
                        joinpath("static/temp/frame.jpg"))
    __path = str(Path(__file__).resolve().parent.parent.parent.
                 joinpath("model"))
    __fd_model = __path + "/face-detection/FP32/face-detection.xml"
    __ag_model = __path + "/age-gender/FP16/age-gender-recognition.xml"
    __em_model = __path + "/emotions/FP32/emotions-recognition.xml"
    __hp_model = __path + "/head-pose/FP32/head-pose-estimation.xml"
    __lm_model = __path + "/facial-landmarks/FP32/facial-landmarks.xml"
    __emotions_list = ["neutro", "feliz", "triste", "surpresa", "raiva"]
    __font = cv2.FONT_HERSHEY_COMPLEX

    def __init__(self, input_stream=0, ag=0, hp=0, em=0, lm=0):
        self._running = True
        self.__age_input_blob, self.__age_out_blob = None, None
        self.__ag_n, self.__ag_c, self.__ag_h = None, None, None
        self.__ag, self.__hp, self.__em, self.__lm = ag, hp, em, lm
        self.__input_stream, self.__frames_counter = input_stream, 0
        self.__plugin, self.__plugin_ag, self.__cap = None, None, None
        self.__n, self.__c, self.__h, self.__w = None, None, None, None
        self.__out_blob, self.__input_blob, self.__exec_net = None, None, None
        self.__age_enabled, self.__age_exec_net, self.__ag_w = None, None, None
        self.__get_video_capture()
        self.util = ApiUtil()

    def start(self):
        # Initializing plugin for CPU device
        self.__load_plugin()
        # Face detection
        self.__face_detection()
        # Age and Gender
        if self.__ag:
            self.__age_and_gender()
        # Head Pose
        if self.__hp:
            self.__head_pose()
        # Emotions, Facial Landmarks
        if self.__em:
            self.__emotions()
            if self.__lm:
                self.__facial_landmarks()
        # ---
        self._running = True
        return self.__starting_inference()

    def terminate(self):
        self._running = False

    def __load_plugin(self):
        # Initializing plugin for device
        self.__plugin = IEPlugin("CPU", None)
        # Loading network files
        self.__plugin.add_cpu_extension(self.__cpu_extension)

    @staticmethod
    def __load_model(model_xml):
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        net = IENetwork(model=model_xml, weights=model_bin)
        return net

    def __get_video_capture(self):
        cap = cv2.VideoCapture(self.__input_stream)
        if not cap.isOpened():
            raise HTTPException(
                status_code=402, detail="Can't open camera by index 1")
        # --
        self.__cur_request_id = 0
        return cap

    def __starting_inference(self):
        cap = self.__get_video_capture()
        while cap and self._running:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break
            # ---
            init_h, init_w, in_frame = self.__change_data_layout(cap, frame)
            # ---
            if self.__exec_net.requests[self.__cur_request_id].wait(-1) == 0:
                # Parse detection results of the current request
                res = self.__exec_net.requests[self.__cur_request_id].outputs[
                    self.__out_blob]
                # Get processing frame
                for obj in res[0][0]:
                    # get objects if probability > specified threshold
                    if obj[2] > self.__prob_threshold:
                        xmin, ymin = int(obj[3] * init_w), int(obj[4] * init_h)
                        xmax, ymax = int(obj[5] * init_w), int(obj[6] * init_h)
                        # Crop the face rectangle for further processing
                        clipped_rect = frame[ymin:ymax, xmin:xmax]
                        if clipped_rect.size == 0:
                            continue

                        height, width = ymax - ymin, xmax - xmin
                        # Age and Gender
                        if self.__ag is 1:
                            self.__age_gender(clipped_rect, frame, xmin, ymin,
                                              xmax, ymax)
                        # Head pose
                        if self.__hp is 1:
                            self.__get_head_pose(clipped_rect, frame, xmin,
                                                 ymin, width, height)
                        # Emotion
                        if self.__em is 1:
                            self.__get_emotions(clipped_rect, frame, xmin,
                                                ymin, width, height)

                        cv2.imwrite(self.__static_path, frame)
                        # yield frame
                        yield (b'--temp\r\n'b'Content-Type: image/jpeg\r\n\r\n'
                               + open(self.__static_path, 'rb').read() +
                               b'\r\n')

    def __face_detection(self):
        net = self.__load_model(self.__fd_model)  # Load Model
        self.__input_blob = next(iter(net.inputs))
        self.__out_blob = next(iter(net.outputs))
        self.__exec_net = self.__plugin.load(network=net, num_requests=2)
        self.__n, self.__c, self.__h, self.__w =\
            net.inputs[self.__input_blob].shape
        del net

    def __age_and_gender(self):
        ag_net = self.__load_model(self.__ag_model)  # Load Model
        self.__age_input_blob = next(iter(ag_net.inputs))
        self.__age_out_blob = next(iter(ag_net.outputs))
        self.__age_exec_net = self.__plugin.load(
            network=ag_net, num_requests=2)
        self.__ag_n, self.__ag_c, self.__ag_h, self.__ag_w = ag_net.inputs[
            self.__input_blob].shape
        del ag_net

    def __head_pose(self):
        hp_net = self.__load_model(self.__hp_model)  # Load Model
        self.__hp_input_blob = next(iter(hp_net.inputs))
        self.__hp_out_blob = next(iter(hp_net.outputs))
        self.__hp_exec_net = self.__plugin.load(network=hp_net, num_requests=2)
        self.__hp_n, self.__hp_c, self.__hp_h, self.__hp_w = hp_net.inputs[
            self.__input_blob].shape
        del hp_net

    def __emotions(self):
        em_net = self.__load_model(self.__em_model)  # Load Model
        self.__em_input_blob = next(iter(em_net.inputs))
        self.__em_out_blob = next(iter(em_net.outputs))
        self.__em_exec_net = self.__plugin.load(network=em_net, num_requests=2)
        self.__em_n, self.__em_c, self.__em_h, self.__em_w = em_net.inputs[
            self.__input_blob].shape
        del em_net

    def __facial_landmarks(self):
        lm_net = self.__load_model(self.__lm_model)  # Load Model
        self.__lm_input_blob = next(iter(lm_net.inputs))
        self.__lm_out_blob = next(iter(lm_net.outputs))
        self.__lm_exec_net = self.__plugin.load(network=lm_net, num_requests=2)
        self.__lm_n, self.__lm_c, self.__lm_h, self.__lm_w = lm_net.inputs[
            self.__input_blob].shape
        del lm_net

    def __change_data_layout(self, cap, frame):
        self.__frames_counter += 1
        init_w = cap.get(3)
        init_h = cap.get(4)
        in_frame = cv2.resize(frame, (self.__w, self.__h))
        # Change data layout from HWC to CHW
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((self.__n, self.__c, self.__h, self.__w))
        self.__exec_net.start_async(request_id=self.__cur_request_id,
                                    inputs={self.__input_blob: in_frame})
        return init_h, init_w, in_frame

    def __age_gender(self, clipped_rect, frame, xmi, ymi, xma, yma):
        clipped_face = cv2.resize(clipped_rect, (self.__ag_w, self.__ag_h))
        # Change data layout from HWC to CHW
        clipped_face = clipped_face.transpose((2, 0, 1))
        clipped_face = clipped_face.reshape((
            self.__ag_n, self.__ag_c, self.__ag_h, self.__ag_w))
        # Get age
        age = self.__get_age(clipped_face)
        prob = ((self.__age_exec_net.requests[
            self.__cur_request_id].outputs['prob'][0][0][0][0]))
        # --
        return self.__print_age_gender(prob, frame, age, xmi, ymi, xma, yma)

    def __get_age(self, clipped_rect):
        self.__age_exec_net.start_async(request_id=0,
                                        inputs={'data': clipped_rect})
        self.__age_exec_net.requests[self.__cur_request_id].wait(-1)
        # --
        return int((self.__age_exec_net.requests[
            self.__cur_request_id].outputs['age_conv3'][0][0][0][0]) * 100)

    def __get_head_pose(self, clipped_rect, frame, xmi, ymi, width, height):
        clipped_face_hp = cv2.resize(clipped_rect, (self.__hp_w, self.__hp_h))
        # Change data layout from HWC to CHW
        clipped_face_hp = clipped_face_hp.transpose((2, 0, 1))
        clipped_face_hp = clipped_face_hp.reshape((self.__hp_n, self.__hp_c,
                                                   self.__hp_h, self.__hp_w))
        self.__hp_exec_net.start_async(
            request_id=0, inputs={'data': clipped_face_hp})
        self.__hp_exec_net.requests[self.__cur_request_id].wait(-1)
        self.__get_pose(frame, xmi, ymi, width, height)

    def __get_pose(self, frame, xmi, ymi, width, height):
        pitch = ((self.__hp_exec_net.requests[self.__cur_request_id].outputs[
            'angle_p_fc'][0][0]))
        yaw = ((self.__hp_exec_net.requests[self.__cur_request_id].outputs[
            'angle_y_fc'][0][0]))
        roll = ((self.__hp_exec_net.requests[self.__cur_request_id].outputs[
            'angle_r_fc'][0][0]))

        c_point = [int(xmi + (width / 2)), int(ymi + (height / 2))]
        self.util.draw_axes(pitch, yaw, roll, c_point, frame)

    def __get_emotions(self, clipped_rect, frame, xmi, ymi, width, height):
        clipped_face_em = cv2.resize(clipped_rect, (self.__em_w, self.__em_h))
        # Change data layout from HWC to CHW
        clipped_face_em = clipped_face_em.transpose((2, 0, 1))
        clipped_face_em = clipped_face_em.reshape((
            self.__em_n, self.__em_c, self.__em_h, self.__em_w))
        self.__em_exec_net.start_async(request_id=0,
                                       inputs={'data': clipped_face_em})
        self.__em_exec_net.requests[self.__cur_request_id].wait(-1)
        # ---
        self.__print_emotions(frame, xmi, ymi)
        # ---
        if self.__lm is 1:
            self.__get_landmarks(clipped_rect, frame, xmi, ymi, width, height)

    def __get_landmarks(self, clipped_rect, frame, xmi, ymi, width, height):
        clipped_face_lm = cv2.resize(clipped_rect, (self.__lm_w, self.__lm_h))
        # Change data layout from HWC to CHW
        clipped_face_lm = clipped_face_lm.transpose((2, 0, 1))
        clipped_face_lm = clipped_face_lm.reshape((
            self.__lm_n, self.__lm_c, self.__lm_h, self.__lm_w))
        self.__lm_exec_net.start_async(request_id=0,
                                       inputs={'data': clipped_face_lm})
        self.__lm_exec_net.requests[self.__cur_request_id].wait(-1)
        # ---
        self.__print_landmarks(frame, xmi, ymi, width, height)

    def __print_emotions(self, frame, xmi, ymi):
        emotion_values = self.__em_exec_net.requests[
            self.__cur_request_id].outputs['prob_emotion']
        emotion_type = emotion_values.argmax()
        result = self.__emotions_list[emotion_type]
        cv2.putText(frame, '   ' + result, (xmi + 40, ymi - 7), self.__font,
                    0.6, (153, 204, 50), 1)

    def __print_landmarks(self, frame, xmi, ymi, width, height):
        for i_lm in range(0, 35):
            normed_x = self.__lm_exec_net.requests[0].outputs[
                'align_fc3'][0][2 * i_lm]
            normed_y = self.__lm_exec_net.requests[0].outputs[
                'align_fc3'][0][(2 * i_lm) + 1]

            x_lm = xmi + width * normed_x
            y_lm = ymi + height * normed_y
            cv2.circle(frame, (int(x_lm), int(y_lm)), 1 + int(0.019 * width),
                       (0, 255, 255), -1)

    def __print_age_gender(self, prob, frame, age, xmi, ymi, xma, yma):
        if prob > 0.5:
            gender = 'F'
            cv2.putText(frame, str(gender) + ',' + str(age) + ',',
                        (xmi, ymi - 7), self.__font, 0.6, (10, 10, 200), 1)
            cv2.rectangle(frame, (xmi, ymi), (xma, yma), (10, 10, 200), 2)
        else:
            gender = 'M'
            cv2.putText(frame, str(gender) + ',' + str(age) + ',',
                        (xmi, ymi - 7), self.__font, 0.6, (10, 10, 200), 1)
            cv2.rectangle(frame, (xmi, ymi), (xma, yma), (255, 10, 10), 2)
