#ifndef CAFFE_YOLOV3_LAYER_HPP_
#define CAFFE_YOLOV3_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <string>
#include "caffe/layers/loss_layer.hpp"
//#include "caffe/layers/region_loss_layer.hpp"
#include <map>
#include "caffe/util/bbox_util.hpp"
namespace caffe {
typedef enum {
  IOU, GIOU, MSE
} IOU_LOSS;  
// box.h

typedef struct dxrep {
    float dt, db, dl, dr;
} dxrep;

// box.h
typedef struct ious {
    float iou, giou;
    dxrep dx_iou;
    dxrep dx_giou;
} ious;
typedef struct boxabs {
  float left, right, top, bot;
} boxabs;

template <typename Dtype>
Dtype overlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2);
template <typename Dtype>
Dtype box_intersection(vector<Dtype> a, vector<Dtype> b);
template <typename Dtype>
Dtype box_union(vector<Dtype> a, vector<Dtype> b);
template <typename Dtype>
Dtype box_iou(vector<Dtype> a, vector<Dtype> b);
template <typename Dtype>
boxabs box_c(vector<Dtype> a, vector<Dtype> b);
template <typename Dtype>
boxabs to_tblr(vector<Dtype> a);
template <typename Dtype>
Dtype box_giou(vector<Dtype> a, vector<Dtype> b);

//template <typename Dtype>
//void get_region_box(vector<Dtype> &b, Dtype* x, vector<Dtype> biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride);
  
struct AvgRegionScore {
	public:
	float avg_anyobj;
	float avg_obj;
	float avg_iou;
	float avg_cat;
	float recall;
	float recall75, loss;
};

template <typename Ftype, typename Btype>
class Yolov3Layer : public LossLayer<Ftype, Btype> {
typedef Ftype Dtype;
public:
  explicit Yolov3Layer(const LayerParameter& param)
    : LossLayer<Ftype, Btype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
    const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
    const vector<Blob*>& top);

  virtual inline const char* type() const { return "Yolov3"; }
  
  class PredictionResult {
  public:
    Dtype x;
    Dtype y;
    Dtype w;
    Dtype h;
    Dtype objScore;
    Dtype classScore;
    Dtype confidence;
    int classType;
  };


  protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top);
  //virtual void Forward_gpu(const vector<Blob*>& bottom,
    //   const vector<Blob*>& top);

  virtual void Backward_cpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  //virtual void Backward_gpu(const vector<Blob*>& top,
    //   const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  int iter_;
  int side_w_;
  int side_h_;
  int num_class_;
  int num_;
  int biases_size_;
  int anchors_scale_;
  int time_count_;
  int class_count_;
  float object_scale_;
  float class_scale_;
  float noobject_scale_;
  float coord_scale_;
  float thresh_;
  bool use_logic_gradient_;
  vector<Dtype> biases_;
  vector<Dtype> mask_;
  //Blob<Dtype> diff_;
  //Blob<Dtype> real_diff_;
  //Blob<Dtype> swap_;
  shared_ptr<Blob> diff_;
  shared_ptr<Blob> real_diff_;
  shared_ptr<Blob> swap_;
  
  AvgRegionScore score_;
  bool use_focal_loss_;
  IOU_LOSS iou_loss_;
  float iou_normalizer_;
};

}  // namespace caffe

#endif  // CAFFE_REGION_LOSS_LAYER_HPP_
