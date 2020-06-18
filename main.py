import setup
import get_frame
import cv2
import ml_model
if __name__ == '__main__':
    # getting name,password,ip,camera number from user
    info = setup.list_maker()
    while True:
        # getting frame from cv2 video capture function using rtsp protocol
        stream_frame = get_frame.get_frame(info)

        # sending frame into out machine learning model
        ml_frame = ml_model.ml_frame(stream_frame)

        # displaying machine learning frame
        cv2.imshow('ml_frame',ml_frame)
        cv2.waitKey(1)
