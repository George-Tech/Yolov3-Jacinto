# Yolov3-Jacinto
Yolov3 layers on Caffe-Jacinto

New Layers:
yolov3_detection_output_layer(for yolov3 detect_out infer) 
yolov3_layer(for yolov3 loss training)

How to use:
1.put annotated_data_layer.cpp | yolov3_detection_output_layer.cpp | yolov3_layer.cpp in $caffe_root/src/caffe/layer
2.put annotated_data_layer.hpp | yolov3_detection_output_layer.hpp | yolov3_layer.hpp in $caffe_root/include/caffe/layer
3.put caffe.proto in $caffe_root/src/caffe/proto
4.make all 

test images:
![image](https://github.com/George-Tech/Misc/blob/master/Images/jacinto-yolov3-test.jpg)
