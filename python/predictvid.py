import sys
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection import ObjectDetection
import cv2
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

MODEL_FILENAME = '../model.pb'
LABELS_FILENAME = '../labels.txt'

class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow
    """
    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
            
    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[:,:,(2,1,0)] # RGB -> BGR
        
        with tf.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis,...]})
            return outputs[0]


def main():
    # Load a TensorFlow model
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(MODEL_FILENAME, 'rb') as f:
        graph_def.ParseFromString(f.read())

    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = TFObjectDetection(graph_def, labels)

    cap = cv2.VideoCapture(0);

    frame_rate = 1
    prev = 0


    while (True):

        time_elapsed = time.time() - prev
        ret, image = cap.read()

        

    # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize=None, fx=0.25, fy=0.25)
    # Convert from array to image, and perform predictions
        if time_elapsed > 1./frame_rate:
            prev = time.time()

            np_im = Image.fromarray(frame)
    
            predictions = od_model.predict_image(np_im)
        # print(predictions)
            for prediction in predictions:
                if prediction['probability'] > 0.5:
                    print(prediction)
            #else: print("low scoring prediction")
        
        
        # Display the resulting frame - will run out of memory on Jetson Nano
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # can perform manipulation here
        # cv2.imshow('frame',frame)
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
    
    cap.release()
    
if __name__ == '__main__':
    main()
