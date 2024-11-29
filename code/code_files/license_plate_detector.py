import cv2
import easyocr
import numpy as np
import matplotlib.pyplot as plt

class LicensePlateDetector:
    def __init__(self, margin=10):
        self.margin = margin  # Margin around detected characters in pixels
        self.color = (0, 255, 0)  # Green color for bounding box
        self.img = None  # Original image
        self.fig_image = None  # Image with detected areas highlighted
        self.reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader

    def detect(self, img_path: str):
        """Detect license plates in the image and extract the text."""
        # Load the image
        self.img = cv2.imread(img_path)
        if self.img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Convert the image to grayscale
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Detect text using EasyOCR
        results = self.reader.readtext(img_gray)

        detected_texts = []  # List to hold the detected license plate texts

        # Draw bounding boxes around the detected text regions and extract text
        for (bbox, text, prob) in results:
            # Extract bounding box coordinates
            (top_left, bottom_right) = bbox[0], bbox[2]
            x1, y1 = int(top_left[0]), int(top_left[1])
            x2, y2 = int(bottom_right[0]), int(bottom_right[1])

            # Add a margin around the detected region
            x1 = max(0, x1 - self.margin)
            y1 = max(0, y1 - self.margin)
            x2 = min(self.img.shape[1], x2 + self.margin)
            y2 = min(self.img.shape[0], y2 + self.margin)

            # Draw the bounding box on the image
            cv2.rectangle(self.img, (x1, y1), (x2, y2), self.color, 2)

            # Append the detected text to the list
            detected_texts.append(text)

        self.fig_image = self.img
        # Return the image with bounding boxes and the detected texts
        return self.img, detected_texts

    def show_detected_image(self):
        """Display the image with detected bounding boxes."""
        if self.fig_image is not None:
            plt.imshow(cv2.cvtColor(self.fig_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        else:
            print("No detected image to display.")

    # def process_video(self, video_path, output_path=None):
    #     """Process video and detect license plates in each frame."""
    #     # Open the video file
    #     cap = cv2.VideoCapture(video_path)

    #     # Get video properties
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     fps = int(cap.get(cv2.CAP_PROP_FPS))
    #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #     # Initialize video writer if saving output
    #     out = None
    #     if output_path:
    #         out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         # Detect license plates in the current frame and extract text
    #         processed_frame, detected_texts = self.detect_in_frame(frame)

    #         # Print the detected text (license plate symbols)
    #         for text in detected_texts:
    #             print("Detected License Plate Text:", text)

    #         # Optionally, show the processed frame
    #         cv2.imshow('License Plate Detection', processed_frame)

    #         # Save the processed frame to the output file if specified
    #         if out:
    #             out.write(processed_frame)

    #         # Break the loop if 'q' is pressed
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    #     # Release resources
    #     cap.release()
    #     if out:
    #         out.release()
    #     cv2.destroyAllWindows()



# # import cv2
# import easyocr
# import numpy as np
# import matplotlib.pyplot as plt

# class LicensePlateDetector:
#     def __init__(self, margin=10):
#         self.margin = margin  # Margin around detected characters in pixels
#         self.color = (0, 255, 0)  # Green color for bounding box
#         self.img = None  # Original image
#         self.fig_image = None  # Image with detected areas highlighted

#     def detect(self, img_path: str):
#         # Load the image
#         self.img = cv2.imread(img_path)
#         if self.img is None:
#             raise FileNotFoundError(f"Image not found at {img_path}")

#         # Convert the image to grayscale
#         img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

#         # Initialize EasyOCR reader
#         reader = easyocr.Reader(['en'])

#         # Detect text using EasyOCR
#         results = reader.readtext(img_gray)

#         # Draw bounding boxes around the detected text regions
#         for (bbox, text, prob) in results:
#             # Extract bounding box coordinates
#             (top_left, bottom_right) = bbox[0], bbox[2]
#             x1, y1 = int(top_left[0]), int(top_left[1])
#             x2, y2 = int(bottom_right[0]), int(bottom_right[1])

#             # Add a margin around the detected region
#             x1 = max(0, x1 - self.margin)
#             y1 = max(0, y1 - self.margin)
#             x2 = min(self.img.shape[1], x2 + self.margin)
#             y2 = min(self.img.shape[0], y2 + self.margin)

#             # Draw the bounding box on the original image
#             cv2.rectangle(self.img, (x1, y1), (x2, y2), self.color, 2)

#         self.fig_image = self.img
#         return self.img

#     def show_detected_image(self):
#         if self.fig_image is not None:
#             plt.imshow(cv2.cvtColor(self.fig_image, cv2.COLOR_BGR2RGB))
#             plt.axis('off')
#             plt.show()
#         else:
#             print("No detected image to display.")

#     def process_video(self, video_path, output_path=None):
#         """Process video and detect license plates in each frame."""
#         # Open the video file
#         cap = cv2.VideoCapture(video_path)

#         # Get video properties
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         # Initialize video writer if saving output
#         out = None
#         if output_path:
#             out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Detect license plates in the current frame
#             processed_frame = self.detect_in_frame(frame)

#             # Save the processed frame to the output file if specified
#             if out:
#                 out.write(processed_frame)

#             # Optionally print or log progress, or add custom handling here if needed
#             # For now, we'll skip cv2.imshow to avoid the error

#         # Release resources
#         cap.release()
#         if out:
#             out.release()
#         # Comment out cv2.destroyAllWindows()
#         # cv2.destroyAllWindows()
