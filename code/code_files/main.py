
import cv2
import easyocr

import numpy as np
import matplotlib.pyplot as plt
from license_plate_detector import LicensePlateDetector



detector = LicensePlateDetector(margin=10)
detected_image = detector.detect("../../images/license.jpg") 
detector.show_detected_image()

detected_image = detector.detect("../../images/license_blue_car.jpg") 
detector.show_detected_image()

detected_image = detector.detect("../../images/no_licens.jpg") 
detector.show_detected_image()

detected_image = detector.detect("../../images/no_license_2.jpg") 
detector.show_detected_image()