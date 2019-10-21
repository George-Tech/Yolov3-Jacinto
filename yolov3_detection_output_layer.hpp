#ifndef CAFFE_YOLOV3_DETECTION_OUTPUT_LAYER_HPP_
#define CAFFE_YOLOV3_DETECTION_OUTPUT_LAYER_HPP_



#include <map>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/bbox_util.hpp"


namespace caffe {
template <typename Dtype>
class _PredictionResult {
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
struct AvgRegionScore {
public:
	float avg_anyobj;
	float avg_obj;
	float avg_iou;
	float avg_cat;
	float recall;
	float recall75, loss;
};

/**
 * @brief Generate the detection output based on location and confidence
 * predictions by doing non maximum suppression.
 *
 * Intended for use with MultiBox detection method.
 *
 * NOTE: does not implement Backwards operation.
 */
//template <typename Dtype>
template <typename Ftype, typename Btype>
class Yolov3DetectionOutputLayer : public Layer<Ftype, Btype> {
 typedef Ftype Dtype;
 public:
  explicit Yolov3DetectionOutputLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "DetectionOutput"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  //virtual inline int MaxBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  void correct_yolo_boxes(_PredictionResult<Dtype> &det, int w, int h, int netw, int neth, int relative);
 protected:
  /**
   * @brief Do non maximum suppression (nms) on prediction results.
   *
   * @param bottom input Blob vector (at least 2)
   *   -# @f$ (N \times C1 \times 1 \times 1) @f$
   *      the location predictions with C1 predictions.
   *   -# @f$ (N \times C2 \times 1 \times 1) @f$
   *      the confidence predictions with C2 predictions.
   *   -# @f$ (N \times 2 \times C3 \times 1) @f$
   *      the prior bounding boxes with C3 values.
   * @param top output Blob vector (length 1)
   *   -# @f$ (1 \times 1 \times N \times 7) @f$
   *      N is the number of detections after nms, and each row is:
   *      [image_id, label, confidence, xmin, ymin, xmax, ymax]
   */
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  //virtual void Forward_gpu(const vector<Blob*>& bottom,
	//  const vector<Blob*>& top);
  /// @brief Not implemented
  virtual void Backward_cpu(const vector<Blob*>& top,
	  const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  //virtual void Backward_gpu(const vector<Blob*>& top,
	//  const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  int side_w_;
  int side_h_;
  int num_class_;
  int num_;
  int coords_;
  
  int mask_group_num_;
  int groups_num_;
  vector< _PredictionResult<Dtype> > predicts_;
  Dtype confidence_threshold_;
  Dtype nms_threshold_;
  vector<Dtype> biases_;
  vector<Dtype> anchors_scale_;
  vector<Dtype> mask_;
  //Blob<Dtype> swap_;
  shared_ptr<Blob> swap_;

};

}  // namespace caffe

#endif  // CAFFE_DETECTION_OUTPUT_LAYER_HPP_
