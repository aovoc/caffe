#include <algorithm>
#include <vector>

#include "caffe/layers/hswish_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void HSwishForward(const int n, const Dtype* in, Dtype* out/*, Dtype negative_slope*/) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > Dtype(-3.) ? in[index] : 0;
    out[index] = out[index]  < Dtype(3.) ? out[index] : Dtype(6.);
    out[index] = in[index] * out[index] /  Dtype(6.);
  }
}

template <typename Dtype>
void HSwishLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  //Dtype negative_slope = this->layer_param_.hswish_param().negative_slope();
  // NOLINT_NEXT_LINE(whitespace/operators)
  HSwishForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);//, negative_slope);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void HSwishBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff/*, Dtype negative_slope*/) {
  CUDA_KERNEL_LOOP(index, n) {

    out_diff[index] = in_diff[index] / Dtype(6.)  * \
    ((((in_data[index]+ Dtype(3.)) * (in_data[index]+ Dtype(3.) > 0))*(in_data[index] < Dtype(3.))) + ( Dtype(6.) * (in_data[index] >= Dtype(3.))));
     //std::min(std::max(in_data[index] + Dtype(3.), Dtype(0)),Dtype(6.))  + (in_diff[index] * in_data[index] / Dtype(6.) * (in_data[index]+ Dtype(3.) > //0))*(in_data[index] < Dtype(3.));

//    out_diff[index] = in_diff[index] * ((in_data[index] > 0)        + (in_data[index] <= 0) * negative_slope) * (in_data[index] < Dtype(6.));

  }
}

template <typename Dtype>
void HSwishLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    //Dtype negative_slope = this->layer_param_.hswish_param().negative_slope();
    // NOLINT_NEXT_LINE(whitespace/operators)
    HSwishBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);//, negative_slope);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(HSwishLayer);


}  // namespace caffe
