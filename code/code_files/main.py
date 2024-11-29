import cv2
from license_plate_detector import LicensePlateDetector

detector = LicensePlateDetector(margin=10)

detected_image,detected_text = detector.detect("../../images/license.jpg")
print("Detected License Plate Texts:", detected_text)  
detector.show_detected_image()
detected_image ,detected_text= detector.detect("../../images/license_blue_car.jpg")
print("Detected License Plate Texts:", detected_text)  
detector.show_detected_image()

detected_image,detected_text = detector.detect("../../images/no_licens.jpg")
print("Detected License Plate Texts:", detected_text)  
detector.show_detected_image()

detected_image ,detected_text= detector.detect("../../images/no_license_2.jpg")
print("Detected License Plate Texts:", detected_text) 
detector.show_detected_image()

# video_path = "../../videos/output_vide_car.mp4"  
# output_video_path = "../../videos/output_license_detection.avi"
# detector.process_video(video_path, output_video_path)
