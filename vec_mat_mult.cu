/* Vector-matrix multiplication: Y = A * X.
 * Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <vec_mat_mult_kernel.cu>

#define MIN_NUMBER 1
#define MAX_NUMBER 4

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void compute_gold(float*, const float*, const float*, unsigned int, unsigned int);

Matrix allocate_matrix_on_gpu(const Matrix M);
Matrix allocate_matrix(int num_rows, int num_columns, int init);
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost);
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice);
void vec_mat_mult_on_device_using_global_memory(const Matrix M, const Matrix N, Matrix P);
void vec_mat_mult_on_device_using_shared_memory(const Matrix M, const Matrix N, Matrix P);
void print_matrix(const Matrix M);
float get_random_number(int, int);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	// Matrices for the program
	Matrix  A; // 4096 x 4096 matrix
	Matrix  X; // 4096 x 1 vector
	Matrix  Y1, Y2; // 4096 x 1 vector
	
	// Initialize the random number generator with a seed value 
	srand(time(NULL));
	
	// Check command line arguments
	if(argc > 1){
		printf("Error. This program accepts no arguments. \n");
		exit(0);
	}		
	 
	// Allocate and initialize the matrices
	A  = allocate_matrix(MATRIX_SIZE, MATRIX_SIZE, 1); // Create a random 512 X 512 matrix
	X  = allocate_matrix(MATRIX_SIZE, 1, 1); // Create a random 512 x 1 vector 
	Y1  = allocate_matrix(MATRIX_SIZE, 1, 0); // Allocate memory for the output vectors
	Y2 = allocate_matrix(MATRIX_SIZE, 1, 0); 
 
	
	// Perform the vector-matrix multiplication on the GPU using global memory
	vec_mat_mult_on_device_using_global_memory(A, X, Y1);
	
	// Perform the vector-matrix multiplication on the GPU using shared memory
	vec_mat_mult_on_device_using_shared_memory(A, X, Y2);
	
	// compute the vector-matrix multiplication on the CPU for comparison
	Matrix reference = allocate_matrix(MATRIX_SIZE, 1, 0);
	
	unsigned int timer;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
	
	compute_gold(reference.elements, A.elements, X.elements, A.num_rows, A.num_columns);
	
	cutStopTimer(timer);
	
	/* print out timing results */
	printf("Kernel execution time on the CPU: %f seconds. \n", (float)cutGetTimerValue(timer)/1000.0);

	// check if the device result is equivalent to the expected solution ( global memory )
	int size_elements = NUM_ROWS;
	CUTBoolean res_global = cutComparefe(reference.elements, Y1.elements, size_elements, 0.0001f);
	printf("CUDA w/ Global Mem Test %s\n", (1 == res_global) ? "PASSED" : "FAILED");

	// check if the device result is equivalent to the expected solution ( shared memory )
	CUTBoolean res_shared = cutComparefe(reference.elements, Y2.elements, size_elements, 0.0001f);
	printf("CUDA w/ Shared Mem Test %s\n", (1 == res_shared) ? "PASSED" : "FAILED");

	// Free host matrices
	free(A.elements); A.elements = NULL;
	free(X.elements); X.elements = NULL;
	free(Y1.elements); Y1.elements = NULL;
	free(Y2.elements); Y2.elements = NULL;
	
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void vec_mat_mult_on_device_using_global_memory(const Matrix A, const Matrix X, Matrix Y){
	//Interface host call to the device kernel code and invoke the kernel	
	
	/* allocate space on gpu */
	Matrix gpu_a = allocate_matrix_on_gpu( A );
	Matrix gpu_x = allocate_matrix_on_gpu( X );
	Matrix gpu_y = allocate_matrix_on_gpu( Y );

	/* copy matrices to gpu  */
	copy_matrix_to_device( gpu_a, A );
	copy_matrix_to_device( gpu_x, X );

	/* set gpu parameters */
	dim3 dimBlock( 512, 1 );
	dim3 dimGrid( MATRIX_SIZE / dimBlock.x, 1 );

	/* init and start timer */
	unsigned int timer;
        cutCreateTimer(&timer);
        cutStartTimer(timer);

	/* invoke kernel on gpu*/
	MatrixMulKernel_using_global_memory <<< dimGrid, dimBlock >>> (gpu_a.elements, gpu_x.elements, gpu_y.elements);

	/* end timer, print result */
	cudaThreadSynchronize();
	cutStopTimer(timer);
	printf("Kernel execution time on GPU (global memory): %f seconds. \n", (float)cutGetTimerValue(timer)/1000.0);

	/* check for gpu errors */
	cudaError_t err =  cudaGetLastError();
	if ( cudaSuccess != err ) {
		fprintf(stderr, " GPU kernel failed: %s.\n", cudaGetErrorString(err));
	}

	/* copy result back to host */
	copy_matrix_from_device( Y, gpu_y );

	/* clean up mem on gpu */
	cudaFree( gpu_a.elements );
	cudaFree( gpu_x.elements );
	cudaFree( gpu_y.elements );
}

void vec_mat_mult_on_device_using_shared_memory(const Matrix A, const Matrix X, Matrix Y){
	//Interface host call to the device kernel code and invoke the kernel	

	/* allocate space on gpu */
        Matrix gpu_a = allocate_matrix_on_gpu( A );
        Matrix gpu_x = allocate_matrix_on_gpu( X );
        Matrix gpu_y = allocate_matrix_on_gpu( Y );

        /* copy matrices to gpu  */
        copy_matrix_to_device( gpu_a, A );
        copy_matrix_to_device( gpu_x, X );

        /* set gpu parameters */
        dim3 dimBlock( 16, 16 );
        dim3 dimGrid( MATRIX_SIZE / dimBlock.x, 1 );

	/* init and start timer */
        unsigned int timer;
        cutCreateTimer(&timer);
        cutStartTimer(timer);

        /* invoke kernel on gpu*/
        MatrixMulKernel_using_shared_memory <<< dimGrid, dimBlock >>> (gpu_a.elements, gpu_x.elements, gpu_y.elements);

	/* end timer, print result */
	cudaThreadSynchronize();
	cutStopTimer(timer);
	printf("Kernel execution time on GPU (shared memory): %f seconds. \n", (float)cutGetTimerValue(timer)/1000.0);
        
	/* check for gpu errors */
        cudaError_t err =  cudaGetLastError();
        if ( cudaSuccess != err ) {
                fprintf(stderr, " GPU kernel failed: %s.\n", cudaGetErrorString(err));
        } 

        /* copy result back to host */
        copy_matrix_from_device( Y, gpu_y );

        /* clean up mem on gpu */
        cudaFree( gpu_a.elements );
        cudaFree( gpu_x.elements );
        cudaFree( gpu_y.elements );
}


// Allocate a device matrix of same size as M.
Matrix allocate_matrix_on_gpu(const Matrix M){
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void**)&Mdevice.elements, size);
    return Mdevice;
}

// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix allocate_matrix(int num_rows, int num_columns, int init){
    	Matrix M;
    	M.num_columns = M.pitch = num_columns;
    	M.num_rows = num_rows;
    	int size = M.num_rows * M.num_columns;
		
	M.elements = (float*) malloc(size*sizeof(float));
	for(unsigned int i = 0; i < size; i++){
		if(init == 0) M.elements[i] = 0; 
		else
			M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    return M;
}	

// Copy a host matrix to a device matrix.
void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice){
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
}

// Prints the matrix out to screen
void print_matrix(const Matrix M){
	for(unsigned int i = 0; i < M.num_rows; i++){
		for(unsigned int j = 0; j < M.num_columns; j++)
			printf("%f ", M.elements[i*M.num_columns + j]);
		printf("\n");
	} 
	printf("\n");
}

// Returns a random floating-point number between the specified min and max values 
float get_random_number(int min, int max){
	return (float)floor((double)(min + (max - min + 1)*((float)rand()/(float)RAND_MAX)));
}


