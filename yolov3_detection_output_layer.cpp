/*
* @Author: Eric612
* @Date:   2018-08-20 
* @https://github.com/eric612/Caffe-YOLOv2-Windows
* @https://github.com/eric612/MobileNet-YOLO
* Avisonic
*/
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>


//#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/layers/yolov3_detection_output_layer.hpp"
#include "caffe/layers/yolov3_conv19.hpp"
#include "caffe/layers/yolov3_conv20.hpp"
#include "caffe/util/io.hpp"


namespace caffe {

template <typename Dtype>
inline Dtype _sigmoid(Dtype x)
{
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
Dtype overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2)
{
  float l1 = x1 - w1 / 2;
  float l2 = x2 - w2 / 2;
  float left = l1 > l2 ? l1 : l2;
  float r1 = x1 + w1 / 2;
  float r2 = x2 + w2 / 2;
  float right = r1 < r2 ? r1 : r2;
  return right - left;
}
template <typename Dtype>
Dtype box_intersection(vector<Dtype> a, vector<Dtype> b)
{
  float w = overlap(a[0], a[2], b[0], b[2]);
  float h = overlap(a[1], a[3], b[1], b[3]);
  if (w < 0 || h < 0) return 0;
  float area = w*h;
  return area;
}
template <typename Dtype>
Dtype box_union(vector<Dtype> a, vector<Dtype> b)
{
  float i = box_intersection(a, b);
  float u = a[2] * a[3] + b[2] * b[3] - i;
  return u;
}
template <typename Dtype>
Dtype box_iou(vector<Dtype> a, vector<Dtype> b)
{
  return box_intersection(a, b) / box_union(a, b);
}
template <typename Dtype>
void setNormalizedBBox(NormalizedBBox& bbox, Dtype x, Dtype y, Dtype w, Dtype h)
{
  Dtype xmin = x - w / 2.0;
  Dtype xmax = x + w / 2.0;
  Dtype ymin = y - h / 2.0;
  Dtype ymax = y + h / 2.0;

  if (xmin < 0.0) {
    xmin = 0.0;
  }
  if (xmax > 1.0) {
    xmax = 1.0;
  }
  if (ymin < 0.0) {
    ymin = 0.0;
  }
  if (ymax > 1.0) {
    ymax = 1.0;
  }
  bbox.set_xmin(xmin);
  bbox.set_ymin(ymin);
  bbox.set_xmax(xmax);
  bbox.set_ymax(ymax);
  float bbox_size = BBoxSize(bbox, true);
  bbox.set_size(bbox_size);
}
template <typename Dtype>
void ApplyNms(vector<_PredictionResult<Dtype> >& boxes, vector<int>& idxes, Dtype threshold) {
  map<int, int> idx_map;
  for (int i = 0; i < boxes.size() - 1; ++i) {
    if (idx_map.find(i) != idx_map.end()) {
      continue;
    }
    for (int j = i + 1; j < boxes.size(); ++j) {
      if (idx_map.find(j) != idx_map.end()) {
        continue;
      }
      vector<Dtype> Bbox1, Bbox2;
      Bbox1.push_back(boxes[i].x);
      Bbox1.push_back(boxes[i].y);
      Bbox1.push_back(boxes[i].w);
      Bbox1.push_back(boxes[i].h);

      Bbox2.push_back(boxes[j].x);
      Bbox2.push_back(boxes[j].y);
      Bbox2.push_back(boxes[j].w);
      Bbox2.push_back(boxes[j].h);

      Dtype iou = box_iou(Bbox1, Bbox2);
      if (iou >= threshold) {
      idx_map[j] = 1;
      }
      /*	NormalizedBBox Bbox1, Bbox2;
      setNormalizedBBox(Bbox1, boxes[i].x, boxes[i].y, boxes[i].w, boxes[i].h);
      setNormalizedBBox(Bbox2, boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h);

      float overlap = JaccardOverlap(Bbox1, Bbox2, true);

      if (overlap >= threshold) {
        idx_map[j] = 1;
      }*/
    }
  }
  for (int i = 0; i < boxes.size(); ++i) {
    if (idx_map.find(i) == idx_map.end()) {
      idxes.push_back(i);
    }
  }
}
template <typename Dtype>
void class_index_and_score(Dtype* input, int classes, _PredictionResult<Dtype>& predict)
{
  Dtype sum = 0;
  Dtype large = input[0];
  int classIndex = 0;
  for (int i = 0; i < classes; ++i) {
    if (input[i] > large)
      large = input[i];
  }
  for (int i = 0; i < classes; ++i) {
    Dtype e = exp(input[i] - large);
    sum += e;
    input[i] = e;
  }

  for (int i = 0; i < classes; ++i) {
    input[i] = input[i] / sum;
  }
  large = input[0];
  classIndex = 0;

  for (int i = 0; i < classes; ++i) {
    if (input[i] > large) {
      large = input[i];
      classIndex = i;
    }
  }
  predict.classType = classIndex;
  predict.classScore = large;
}

template <typename Dtype>
void get_region_box(vector<Dtype> &b, Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
	b.clear();
	b.push_back((i + x[index + 0 * stride]) / lw);
	b.push_back((j + x[index + 1 * stride]) / lh);
	b.push_back(exp(x[index + 2 * stride]) * biases[2 * n] / (w));
	b.push_back(exp(x[index + 3 * stride]) * biases[2 * n + 1] / (h));
}
template <>
void get_region_box(vector<float> &b, float* x, vector<float> biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
	b.clear();
	b.push_back((i + x[index + 0 * stride]) / lw);
	b.push_back((j + x[index + 1 * stride]) / lh);
	b.push_back(exp(x[index + 2 * stride]) * biases[2 * n] / (w));
	b.push_back(exp(x[index + 3 * stride]) * biases[2 * n + 1] / (h));
}
template <>
void get_region_box(vector<double> &b, double* x, vector<double> biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
	b.clear();
	b.push_back((i + x[index + 0 * stride]) / lw);
	b.push_back((j + x[index + 1 * stride]) / lh);
	b.push_back(exp(x[index + 2 * stride]) * biases[2 * n] / (w));
	b.push_back(exp(x[index + 3 * stride]) * biases[2 * n + 1] / (h));
}



template <typename Ftype, typename Btype>
void Yolov3DetectionOutputLayer<Ftype, Btype>::correct_yolo_boxes(_PredictionResult<Dtype> &det, int w, int h, int netw, int neth, int relative)
{
  //int i;
  int new_w=0;
  int new_h=0;
  if (((float)netw/w) < ((float)neth/h)) {
      new_w = netw;
      new_h = (h * netw)/w;
  } else {
      new_h = neth;
      new_w = (w * neth)/h;
  }
  _PredictionResult<Dtype> &b = det;
  b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
  b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
  b.w *= (float)netw/new_w;
  b.h *= (float)neth/new_h;
  if(!relative){
      b.x *= w;
      b.w *= w;
      b.y *= h;
      b.h *= h;
  }

}

template <typename Ftype, typename Btype>
void Yolov3DetectionOutputLayer<Ftype, Btype>::LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const Yolov3DetectionOutputParameter& yolov3_detection_output_param =
      this->layer_param_.yolov3_detection_output_param();
  CHECK(yolov3_detection_output_param.has_num_classes()) << "Must specify num_classes";
  side_w_ = bottom[0]->width();
  side_h_ = bottom[0]->height();
  num_class_ = yolov3_detection_output_param.num_classes();
  num_ = yolov3_detection_output_param.num_box();
  coords_ = 4;
  confidence_threshold_ = yolov3_detection_output_param.confidence_threshold();
  nms_threshold_ = yolov3_detection_output_param.nms_threshold();
  mask_group_num_ = yolov3_detection_output_param.mask_group_num();
  for (int c = 0; c < yolov3_detection_output_param.biases_size(); ++c) {
     biases_.push_back(yolov3_detection_output_param.biases(c));
  } 
  for (int c = 0; c < yolov3_detection_output_param.mask_size(); ++c) {
    mask_.push_back(yolov3_detection_output_param.mask(c));
  } 
  for (int c = 0; c < yolov3_detection_output_param.anchors_scale_size(); ++c) {
    anchors_scale_.push_back(yolov3_detection_output_param.anchors_scale(c));
  }
  groups_num_ = yolov3_detection_output_param.mask_size() / mask_group_num_;
  swap_ = Blob::create<Dtype>();
  //------------------------------------
  //LOG(INFO)<<"side_w wide_h"<<side_w_<<" "<<side_h_;
  //LOG(INFO)<<"num_class_ "<<num_class_;
  //LOG(INFO)<<"mask_group_num_ "<<mask_group_num_;
  //------------------------------------
  CHECK_EQ(bottom.size(), mask_group_num_);
}

template <typename Ftype, typename Btype>
void Yolov3DetectionOutputLayer<Ftype, Btype>::Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {

  //CHECK_EQ(bottom[0]->num(), 1);
  // num() and channels() are 1.
  vector<int> top_shape(2, 1);
  // Since the number of bboxes to be kept is unknown before nms, we manually
  // set it to (fake) 1.
  top_shape.push_back(1);
  // Each row is a 7 dimension vector, which stores
  // [image_id, label, confidence, x, y, w, h]
  top_shape.push_back(7);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
bool BoxSortDecendScore(const _PredictionResult<Dtype>& box1, const _PredictionResult<Dtype>& box2) {
  return box1.confidence> box2.confidence;
}

template <typename Ftype, typename Btype>
void Yolov3DetectionOutputLayer<Ftype, Btype>::Forward_cpu(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  const int num = bottom[0]->num();
  
  int len = 4 + num_class_ + 1;
  //int stride = side_w_*side_h_;

  //------------------------------------
  //LOG(INFO)<<"LOG1 if_go_to caffe:mode=cpu ";
  //------------------------------------
  if (1) {//(Caffe::mode() == Caffe::CPU) {
    //------------------------------------
	//LOG(INFO)<<"LOG2 go to caffe:mode=cpu ";
    //------------------------------------
    int mask_offset = 0;
    predicts_.clear();
    Dtype *class_score = new Dtype[num_class_];
	//LOG(INFO)<<"bottom size = "<<bottom.size();
    for (int t = 0; t < bottom.size(); t++) {
	  //LOG(INFO)<<"LOG3 DetectLyaerNum "<<t;
      side_w_ = bottom[t]->width();
      side_h_ = bottom[t]->height();
      int stride = side_w_*side_h_;
      swap_->ReshapeLike(*bottom[t]);
      Dtype* swap_data = swap_->mutable_cpu_data<Ftype>();
	  
      const Dtype* input_data = bottom[t]->cpu_data<Ftype>();
	  /*const Dtype* input_data;
	  if (t == 0) {
		input_data = (Dtype*)&conv19_data[0];
	  } else {
	    input_data = (Dtype*)&conv20_data[0];
	  }*/
	  
	  
      int nw = side_w_*anchors_scale_[t];
      int nh = side_w_*anchors_scale_[t];
	  
      for (int b = 0; b < bottom[t]->num(); b++) {
        for (int s = 0; s < side_w_*side_h_; s++) {
          //LOG(INFO) << "LOG4 det_cell_num" <<s;
          for (int n = 0; n < num_; n++) {
            //LOG(INFO) << bottom[t]->count(1);
            int index = n*len*stride + s + b*bottom[t]->count(1);
            vector<Dtype> pred;

            for (int c = 0; c < len; ++c) {
              int index2 = c*stride + index;
              //LOG(INFO)<<index2;

              if (c == 2 || c == 3) {
                swap_data[index2] = (input_data[index2 + 0]);			
              }
              else {
                if (c > 4) {
                  //LOG(INFO) << c - 5;
                  class_score[c - 5] = _sigmoid(input_data[index2 + 0]);
                }
                else {
                  swap_data[index2] = _sigmoid(input_data[index2 + 0]); 
                }
              }
            }
            int y2 = s / side_w_;
            int x2 = s % side_w_;
            Dtype obj_score = swap_data[index + 4 * stride];
            _PredictionResult<Dtype> predict;
            for (int c = 0; c < num_class_; ++c) {
              class_score[c] *= obj_score;
              //LOG(INFO) << class_score[c];
              if (class_score[c] > confidence_threshold_)
              {
                get_region_box(pred, swap_data, biases_, mask_[n + mask_offset], index, x2, y2, side_w_, side_h_, nw, nh, stride);
                predict.x = pred[0];
                predict.y = pred[1];
                predict.w = pred[2];
                predict.h = pred[3];
                predict.classType = c;
                predict.confidence = class_score[c];
                correct_yolo_boxes(predict,side_w_,side_h_,nw,nh,1);
                predicts_.push_back(predict);
              }
            }
          }
        }
      }
      mask_offset += groups_num_;
	  //LOG(INFO)<<"LOG5 mask_offset "<<mask_offset;
    }

    delete[] class_score;
  }
  //LOG(INFO)<<"LOG6 predicts_size "<<predicts_.size();
  std::sort(predicts_.begin(), predicts_.end(), BoxSortDecendScore<Dtype>);
  vector<int> idxes;
  int num_kept = 0;
  if(predicts_.size() > 0){
    //LOG(INFO) << predicts.size();
    ApplyNms(predicts_, idxes, nms_threshold_);
    num_kept = idxes.size();
    //LOG(INFO) << num_kept;
    
  }
  vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  top_shape.push_back(7);

  Dtype* top_data;
  
  if (num_kept == 0) {
    DLOG(INFO) << "Couldn't find any detections";
    top_shape[2] = swap_->num();
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data<Ftype>();
    caffe_set<Dtype>(top[0]->count(), -1, top_data);
    // Generate fake results per image.
    for (int i = 0; i < num; ++i) {
      top_data[0] = i;
      top_data += 7;
    }
  } 
  else {
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data<Ftype>();
    for (int i = 0; i < num_kept; i++){
      top_data[i*7] = 0;                              //Image_Id
      top_data[i*7+1] = predicts_[idxes[i]].classType + 1; //label
      top_data[i*7+2] = predicts_[idxes[i]].confidence; //confidence
      float left = (predicts_[idxes[i]].x - predicts_[idxes[i]].w / 2.);
      float right = (predicts_[idxes[i]].x + predicts_[idxes[i]].w / 2.);
      float top = (predicts_[idxes[i]].y - predicts_[idxes[i]].h / 2.);
      float bot = (predicts_[idxes[i]].y + predicts_[idxes[i]].h / 2.);

      top_data[i*7+3] = left;
      top_data[i*7+4] = top;
      top_data[i*7+5] = right;
      top_data[i*7+6] = bot;
      DLOG(INFO) << "Detection box"  << "," << predicts_[idxes[i]].classType << "," << predicts_[idxes[i]].x << "," << predicts_[idxes[i]].y << "," << predicts_[idxes[i]].w << "," << predicts_[idxes[i]].h;
    }

  }

}
template <typename Ftype, typename Btype>
void Yolov3DetectionOutputLayer<Ftype, Btype>::Backward_cpu(const vector<Blob*>& top,
	const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
	return;
}
#ifdef CPU_ONLY
template <typename Ftype, typename Btype>
void Yolov3DetectionOutputLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
  const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  return;
}
  STUB_GPU_FORWARD(Yolov3DetectionOutputLayer, Forward);
#endif

INSTANTIATE_CLASS_FB(Yolov3DetectionOutputLayer);
REGISTER_LAYER_CLASS(Yolov3DetectionOutput);

}  // namespace caffe
