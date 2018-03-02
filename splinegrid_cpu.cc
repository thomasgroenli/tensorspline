#include "splines.h"

//CPU specialization of actual computation.

/*float kernel_cpu(float x, int n, int dx) {
  float sigmasq = (n+1)/12.;
  
  float a = 1/sqrt(2*M_PI*sigmasq)*exp(-0.5*x*x/sigmasq),b=0;
  for(int n=1; n<=dx; n++) {
    a = -(x*a+(n-1)*b)/sigmasq;
    b = a;
  }
  return a;
}*/

float kernel_cpu(float x, int p, int dx, float *tmp) {
	x += (p + 1) / 2.;
	int k = x;
	for (int i = 0; i < p + 1; i++) {
		tmp[i] = 0;
	}
	tmp[k] = 1;
	for (int i = 0; i < p; i++) {
		for (int j = 0; j < p - i; j++) {
			tmp[j] = i < p-dx ? (x - j) / (i + 1)*tmp[j] + (i + 2 + j - x) / (i + 1)*tmp[j + 1] : tmp[j] - tmp[j + 1];
		}
	}
	return tmp[0];
}

void spline_grid_kernel_cpu(int N, int ndims, int n_neigh, int channels, const int *grid_dim, const int *strides, const int *K, const int *dx, const float *positions, const float *coefficients, float *out) {
    int *idx = new int[ndims];
    float *shift = new float[ndims];
    float *channel_sum = new float[channels];
	int max_order = 0;
	for (int i = 0; i < ndims; i++) {
		max_order = K[i] > max_order ? K[i] : max_order;
	}
	float *kernel_tmp = new float[max_order+1];
    
    float tmp;
    int reduce;
    int flat;
    float Wij;
    for(int i=0; i<N; i++) {
		bool valid = true;
		for(int j=0; j<ndims; j++) {
			tmp = positions[i*ndims+j];
			valid &= (0 <= tmp && tmp < 1);
			shift[j] = modff(tmp*(grid_dim[j]-1)+0.5,&tmp)-0.5;
			idx[j] = tmp;
		}
		for(int j=0; j<channels; j++) {
			channel_sum[j] = 0;
		}

		for(int j=0; j<n_neigh; j++) {
			reduce = j;
			flat = 0;
			Wij = 1;
			for(int k=ndims-1; k>=0; k--) {
				int offset = - (K[k] + 1 - int(shift[k] + 1)) / 2 + (reduce % (K[k] + 1));
				int current = idx[k] + offset;
				float x = shift[k] - offset;
				current = fmin(fmax(current, 0), grid_dim[k] - 1);

				flat += strides[k]*current;
				Wij *= kernel_cpu(x,K[k],dx[k],kernel_tmp);
				reduce/=K[k]+1;
			}
			for(int k=0; k<channels; k++) {
				channel_sum[k] += Wij*(valid?coefficients[channels*flat+k]:NAN);
			}
		}
		for(int j=0; j<channels; j++) {
			out[i*channels+j] = channel_sum[j];
		}
    }
    delete [] idx, shift, channel_sum, kernel_tmp;
}

template<typename T>
struct SplineGridFunctor<Eigen::ThreadPoolDevice, T> {
  void operator()(const Eigen::ThreadPoolDevice& d, const Grid &grid, int N, const float *positions, const float *coefficients, float *out) {

    int ndims = grid.ndims();
    int n_neigh = grid.neighbors();
    int channels = grid.channels;
    std::vector<int> strides = grid.strides();
    std::vector<int> grid_dim = grid.dims;
    std::vector<int> K = grid.K;
    std::vector<int> dx = grid.dx;
	grid.debug();
    spline_grid_kernel_cpu(N, ndims, n_neigh, channels, grid_dim.data(), strides.data(), K.data(), dx.data(), positions, coefficients, out);

  }
     
};



template struct SplineGridFunctor<Eigen::ThreadPoolDevice, float>;

