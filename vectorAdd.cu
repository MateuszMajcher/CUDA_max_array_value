#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <cuda_runtime.h>
#include <math_constants.h>

using namespace std;
__constant__  int ROWS;

/*spr bledow*/
void checkError(cudaError_t err, char* message) {
	if (err != cudaSuccess)
	{
		fprintf(stderr, message, cudaGetErrorString(err));
		//exit(EXIT_FAILURE);
	}
}

/*ustawienie pos dla 2D tablicy*/
template<typename T>
void setElement(T* arr, int width, int row, int col, T value) {
	arr[width * row + col] = value;
}

/*pobranie elem dla 2D tablicy*/
template<typename T>
T getElement(T* arr, int width, int row, int col) {
	return arr[row * width + col];
}

/*inicjalizacja tablicy 2D*/
template<typename T>
void initArray2D(T* arr, int rows, int cols, T value) {
	int c = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			setElement(arr, cols, i, j, rand() % 10);
		}
	}
}

/*wyswietlanie tablicy 2D*/
template<typename T>
void displayArray2D(T* arr, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		printf("\n");
		for (int j = 0; j < cols; j++) {
			cout << getElement(arr, cols, i, j) << " ";
		}
	}
}

/*wyswietlanie tablicy 1D*/
template<typename T>
void displayArray(T* arr, int size) {
	for (int i = 0; i < size; i++) {
		cout << arr[i] << endl;
	}
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
	int g_id = (blockDim.x * blockIdx.x) + tid;

	cache[tid] = -FLT_MAX;

	if (gid < size)
		cache[tid] = src[gid];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s && gid < size)
			cache[tid] = max(cache[tid], cache[tid + s]);  // 2
		__syncthreads();
	}

	if (tid == 0)
		dst[blockIdx.x] = cache[0];
}

int main(int argc, char **argv) {

	int N = 100000;
	float *data = new float[N];
	size_t data_size = N * sizeof(float);



	float min = 0, d_min = 0;

	for (size_t i = 0; i < N; ++i) {
		data[i] = rand()%512512;
		//std::cout << data[i] << std::endl;
		min = fmax(min, data[i]);
	}

	float *dSrc, *dDst;
	cudaError_t err;

	err = cudaMalloc(&dSrc, data_size);
	//check_error(err, "allocating array");

	err = cudaMemcpy(dSrc, data, data_size, cudaMemcpyHostToDevice);
	//check_error(err, "copy UP");

	
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	cout <<"liczba blokow<< "<< blocksPerGrid << endl;

	err = cudaMalloc(&dDst, threadsPerBlock*sizeof(float));
	//check_error(err, "allocating Dst array");
	// Check for errors.
	err = cudaMemcpyToSymbol(ROWS, &N, sizeof(int));
	

	findMin << < blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(float) >> >(dDst, dSrc, N);
	findMin << <1, threadsPerBlock, threadsPerBlock*sizeof(float) >> >(dDst, dDst, blocksPerGrid);

	size_t local_work_size = 256;
	size_t global_work_size = computeGlobalWorkSize(N, local_work_size);
	int num_work_groups = global_work_size / local_work_size;


	cout << "global_work_size work size: " << global_work_size << endl;
	cout << "num work groups: " << blocksPerGrid << endl;



	int step = 0;
	while (num_work_groups > 0)
	{
		cout << "Step " << ++step << endl;
		cout << "Size of data to be reduced: " << N << endl;
		cout << "Local work size: " << local_work_size << endl;
		cout << "Global work size: " << global_work_size << endl;
		cout << "Num of work-groups: " << num_work_groups << endl << endl;
		
	}

	





	err = cudaMemcpy(&d_min, dDst, sizeof(d_min), cudaMemcpyDeviceToHost);
	

	cudaFree(dSrc); dSrc = NULL;
	cudaFree(dDst); dDst = NULL;
	free(data);

	printf("Parallel min: %g vs %g\n", d_min, min);
	system("pause");
}