#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <cuda_runtime.h>
#include <math_constants.h>

using namespace std;
__constant__  int ROWS;

/*check error*/
void checkError(cudaError_t err, char* message) {
	if (err != cudaSuccess)
	{
		fprintf(stderr, message, cudaGetErrorString(err));
		//exit(EXIT_FAILURE);
	}
}

/*set value */
template<typename T>
void setElement(T* arr, int width, int row, int col, T value) {
	arr[width * row + col] = value;
}

/*get value*/
template<typename T>
T getElement(T* arr, int width, int row, int col) {
	return arr[row * width + col];
}

/*Fill an array 2D with values*/
template<typename T>
void initArray2D(T* arr, int rows, int cols, T value) {
	int c = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			setElement(arr, cols, i, j, rand() % 10);
		}
	}
}

/*show array 2D*/
template<typename T>
void displayArray2D(T* arr, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		printf("\n");
		for (int j = 0; j < cols; j++) {
			cout << getElement(arr, cols, i, j) << " ";
		}
	}
}

/*show array 1D*/
template<typename T>
void displayArray(T* arr, int size) {
	for (int i = 0; i < size; i++) {
		cout << arr[i] << endl;
	}
}

/*rand float*/
float randomFloat(float min, float max) {
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = max - min;
	float r = random * diff;
	return min + r;
}

int computeGlobalWorkSize(int dataSize, int localWorkSize)
{
	return (dataSize%localWorkSize) ? dataSize - dataSize%localWorkSize +
	localWorkSize : dataSize;
}

__global__ void findMin(float *dst, const float *src, int size)
{
	extern volatile __shared__ float cache[];

	int l_id = threadIdx.x;
	int g_id = (blockDim.x * blockIdx.x) + l_id;



	if (g_id < size)
		cache[l_id] = src[g_id];
	else
		cache[l_id] = -FLT_MAX;
	
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (l_id < s && l_id < size)
			cache[l_id] = max(cache[l_id], cache[l_id + s]);  // 2
		__syncthreads();
	}

	if (l_id == 0)
		dst[blockIdx.x] = cache[0];

}

int main(int argc, char **argv) {

	int N = 100000;
	float *data = new float[N];
	size_t data_size = N * sizeof(float);



	float min = 0, d_min = 0;

	for (size_t i = 0; i < N; ++i) {
		data[i] = randomFloat(0, 10);
		//check cpu
		min = fmax(min, data[i]);
	}

	float *dSrc, *dDst;
	cudaError_t err;

	err = cudaMalloc(&dSrc, data_size);
	checkError(err, "blad alokacji");

	err = cudaMemcpy(dSrc, data, data_size, cudaMemcpyHostToDevice);
	checkError(err, "blad alokacji");

	
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	cout <<"liczba blokow<< "<< blocksPerGrid << endl;

	err = cudaMalloc(&dDst, threadsPerBlock*sizeof(float));
	checkError(err, "blad alokacji dDst");
	
	err = cudaMemcpyToSymbol(ROWS, &N, sizeof(int));
	checkError(err, "blad alokacji ROWS");



	size_t local_work_size = 256;
	size_t global_work_size = computeGlobalWorkSize(N, local_work_size);
	int num_work_groups = global_work_size / local_work_size;


	int step = 0;
	while (num_work_groups > 0)
	{
		cout << "Step " << ++step << endl;
		cout << "Size of data to be reduced: " << N << endl;
		cout << "Local work size: " << local_work_size << endl;
		cout << "Global work size: " << global_work_size << endl;
		cout << "Num of work-groups: " << num_work_groups << endl << endl;
		
		findMin << < num_work_groups, local_work_size, local_work_size*sizeof(float) >> >(dDst, dSrc, N);
		err = cudaDeviceSynchronize();
		checkError(err, "blad synchronizacji kernala");

		if (num_work_groups > 1)
		{
			N = num_work_groups;
			global_work_size = computeGlobalWorkSize(N, local_work_size);
			num_work_groups = global_work_size / local_work_size;
			float* tmp = dDst;
			dSrc = dDst;
			dDst = tmp;
		}
		else
			num_work_groups = 0;

		err = cudaMemcpy(&d_min, dDst, sizeof(d_min), cudaMemcpyDeviceToHost);
		printf("Parallel min: %g \n", d_min);
	}

	

	err = cudaMemcpy(&d_min, dDst, sizeof(d_min), cudaMemcpyDeviceToHost);
	printf("Parallel min: GPU: %g vs CPU: %g\n", d_min, min);

	cudaFree(dSrc);
	dSrc = NULL;
	cudaFree(dDst); 
	dDst = NULL;
	free(data);

	
	system("pause");
}