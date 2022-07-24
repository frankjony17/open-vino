import math

import cv2
import numpy as np


class ApiUtil:

    __cv_pi = 3.1415926535897932384626433832795

    def draw_axes(self, pitch, yaw, roll, c_point, frame):
        pitch *= self.__cv_pi / 180.0
        yaw *= self.__cv_pi / 180.0
        roll *= self.__cv_pi / 180.0

        yaw_matrix = np.matrix([[math.cos(yaw), 0, -math.sin(yaw)], [
            0, 1, 0], [math.sin(yaw), 0, math.cos(yaw)]])

        pitch_matrix = np.matrix([[1, 0, 0], [
            0, math.cos(pitch), -math.sin(pitch)], [
            0, math.sin(pitch), math.cos(pitch)]])

        roll_matrix = np.matrix([[math.cos(roll), -math.sin(roll), 0], [
            math.sin(roll), math.cos(roll), 0], [0, 0, 1]])

        # Rotational Matrix
        r = yaw_matrix * pitch_matrix * roll_matrix
        rows = frame.shape[0]
        cols = frame.shape[1]

        camera_matrix = np.zeros((3, 3), dtype=np.float32)
        camera_matrix[0][0] = 950.0
        camera_matrix[0][2] = cols / 2
        camera_matrix[1][0] = 950.0
        camera_matrix[1][1] = rows / 2
        camera_matrix[2][1] = 1

        x_axis = np.zeros((3, 1), dtype=np.float32)
        x_axis[0] = 50
        x_axis[1] = 0
        x_axis[2] = 0

        y_axis = np.zeros((3, 1), dtype=np.float32)
        y_axis[0] = 0
        y_axis[1] = -50
        y_axis[2] = 0

        z_axis = np.zeros((3, 1), dtype=np.float32)
        z_axis[0] = 0
        z_axis[1] = 0
        z_axis[2] = -50

        z_axis_1 = np.zeros((3, 1), dtype=np.float32)
        z_axis_1[0] = 0
        z_axis_1[1] = 0
        z_axis_1[2] = 50

        o = np.zeros((3, 1), dtype=np.float32)
        o[2] = camera_matrix[0][0]

        x_axis = r * x_axis + o
        y_axis = r * y_axis + o
        z_axis = r * z_axis + o
        z_axis_1 = r * z_axis_1 + o

        p2x = int((x_axis[0] / x_axis[2] * camera_matrix[0][0]) + c_point[0])
        p2y = int((x_axis[1] / x_axis[2] * camera_matrix[1][0]) + c_point[1])
        cv2.line(frame, (c_point[0], c_point[1]), (p2x, p2y), (0, 0, 255), 2)

        p2x = int((y_axis[0] / y_axis[2] * camera_matrix[0][0]) + c_point[0])
        p2y = int((y_axis[1] / y_axis[2] * camera_matrix[1][0]) + c_point[1])
        cv2.line(frame, (c_point[0], c_point[1]), (p2x, p2y), (0, 255, 0), 2)

        p1x = int((z_axis_1[0] / z_axis_1[2] * camera_matrix[0][0])
                  + c_point[0])
        p1y = int((z_axis_1[1] / z_axis_1[2] * camera_matrix[1][0])
                  + c_point[1])

        p2x = int((z_axis[0] / z_axis[2] * camera_matrix[0][0]) + c_point[0])
        p2y = int((z_axis[1] / z_axis[2] * camera_matrix[1][0]) + c_point[1])

        cv2.line(frame, (p1x, p1y), (p2x, p2y), (255, 0, 0), 2)
        cv2.circle(frame, (p2x, p2y), 3, (255, 0, 0))
