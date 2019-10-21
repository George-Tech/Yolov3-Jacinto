../../build/tools/caffe train -solver="mobilenet_yolov3_lite_solver.prototxt" \
-weights="mobilenet_yolov3_bn_lite__iter_1000.caffemodel" \
-gpu 1 2>&1 | tee log_0827.log
