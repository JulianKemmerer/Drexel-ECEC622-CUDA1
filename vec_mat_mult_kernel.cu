/* Vector-Matrix multiplication: Y = A * X.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "vec_mat_mult.h"


////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel_using_global_memory(float *Ad, float *Xd, float *Yd)
{
	//Multiply A nd X

	/* thread id */
	int tx = blockIdx.x * blockDim.x + threadIdx.x;

	/* variable for partial product */
	float prod = 0;

	/* calculation loop - row of A X col of X */
	for ( int i = 0; i < MATRIX_SIZE; ++i ) {
		
		float A_element = Ad[ MATRIX_SIZE*tx + i ];
		float X_element = Xd[ i ];
		prod += A_element * X_element;
	}

	/* store result */
	Yd[ tx ] = prod;
}


__global__ void MatrixMulKernel_using_shared_memory(float *Ad, float *Xd, float *Yd)
{
	//Multiply A nd X

	/* declare shared memory */
	__shared__ float shared_X[ 16 ];
	__shared__ float shared_A[ 16 ][ 16 ];

	/* thread id */
	int row_num = blockIdx.y * blockDim.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	/* variable for partial product */
	int temp = 0;

	/* calculation loop -> uses memory coalescing */
	for ( int i = 0; i < MATRIX_SIZE; i = i + 16) {

		/* transfers from global to shared memory -> coalesced */
		shared_A[ ty ][ tx ] = Ad[ MATRIX_SIZE * row_num + tx  + i ];
		shared_X[ tx ] = Xd[ tx + i ];

		__syncthreads();

		/* only first thread in row does actual calculation */
		if ( threadIdx.x == 0 ) {
		
			for ( int k = 0; k < blockDim.x; k++ ) {
				temp += shared_A[ tx ][ k ] * shared_X[k];
			}


		}
		__syncthreads();
	}
	
	/* only have first thread in row report */
	if ( threadIdx.x == 0 ){
		Yd[ row_num ] = temp;
	}
}


#endif // #ifndef _MATRIXMUL_KERNEL_H_
