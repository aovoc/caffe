#include <algorithm>
#include <vector>

#include "caffe/layers/hsigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
void HSigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
      const Dtype* bottom_data = bottom[0]->cpu_data();
      Dtype* top_data = top[0]->mutable_cpu_data();
      const int count = bottom[0]->count();
      for (int i = 0; i < count; ++i) {
        top_data[i] = std::min(std::max(bottom_data[i] + Dtype(3.), Dtype(0)),Dtype(6.)) / Dtype(6.);
      }
}

template <typename Dtype>
void HSigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] / Dtype(6.) * (bottom_data[i] + Dtype(3.)> 0)*(bottom_data[i] + Dtype(3.) < Dtype(6.));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(HSigmoidLayer);
#endif

INSTANTIATE_CLASS(HSigmoidLayer);
REGISTER_LAYER_CLASS(HSigmoid);
}  // namespace caffe
