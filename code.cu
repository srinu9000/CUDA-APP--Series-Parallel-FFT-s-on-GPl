#include<sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define PI M_PI
#define _USE_MATH_DEFINES
#define TWOPI	(2.0*PI)
#include <complex.h>
#include <iostream>
#include <cuComplex.h>
#define accuracy 0.001
#include<time.h>
using namespace std;



//#define N 16
//#define N 512
//#define N 1024
//#define N 2048
//#define N 4096
//#define N 16384
//#define N 65536
#define N 131072
//#define N 262144
//#define N 524288
//#define N 1048576
//#define N 2097152
//#define N 4194304
//#define N 8388608
// creating structure for parallel FFT C code
//typedef complex cuFloatComplex;	
typedef struct complex_t {
	float re;
	float im;
}t;

long countE = 0;

t complex_from_polar(double r, double theta_radians) {
	t result;
	result.re = r * cos(theta_radians);
	result.im = r * sin(theta_radians);
	return result;
}

double complex_magnitude(t c) {
	return sqrt(c.re*c.re + c.im*c.im);
}

t complex_add(t left, t right) {
	t result;
	result.re = left.re + right.re;
	result.im = left.im + right.im;
	return result;
}

t complex_sub(t left, t right) {
	t result;
	result.re = left.re - right.re;
	result.im = left.im - right.im;
	return result;
}

t complex_mult(t left,t right) {
	t result;
	result.re = left.re*right.re - left.im*right.im;
	result.im = left.re*right.im + left.im*right.re;
	return result;
}

complex_t* wnn = (t *)malloc(N*sizeof(t));

/* structure complete*/



/* CUDA wn kernel*/
__global__ void W(cuFloatComplex *w_out)
{
	long k =blockIdx.x*blockDim.x+threadIdx.x;
	w_out[k]=make_cuFloatComplex(cos(-2.0*PI*k/N),sin(-2.0*PI*k/N));
	//w_out[k]=make_cuFloatComplex(cuCrealf(WnS[k]),cuCimagf(WnS[k]));
	__syncthreads();
}

/*CUDA FFT kernel*/
__global__ void FFT(cuFloatComplex *d_out, cuFloatComplex *d_in, long stage, long Bsize, long NeachL,cuFloatComplex *WnS)
{

	long k = NeachL*(blockIdx.x*blockDim.x+threadIdx.x);
 	
	//cuFloatComplex Wn; 
	if(k<N){		
		//butterfly
		for(long as = 0; as<Bsize;as++){
			if(k+as+NeachL/2<N){

				d_out[k+as]=cuCaddf(d_in[k+as],cuCmulf(WnS[as*N/NeachL],d_in[k+NeachL/2+as]));
				d_out[k+as+NeachL/2]=cuCsubf(d_in[k+as],cuCmulf(WnS[as*N/NeachL],d_in[k+as+NeachL/2]));
__syncthreads();				
d_in[k+as]=make_cuFloatComplex(cuCrealf(d_out[k+as]),cuCimagf(d_out[k+as]));
				d_in[k+as+NeachL/2]=make_cuFloatComplex(cuCrealf(d_out[k+as+NeachL/2]),cuCimagf(d_out[k+as+NeachL/2]));
									
			}	
		}

	}
}

/*serial DFT*/
t* DFT(t* x) {
t* X1 = (t*) malloc(sizeof(struct complex_t)*N);
long k,n;
for(k=0;k<N;k++)
{
X1[k].re = X1[k].im = 0.0;
for(n=0;n<N;n++)
{
X1[k] = complex_add(X1[k], complex_mult(x[n], complex_from_polar(1,-2*PI*n*k/N)));
}
}
return X1;
}


/* simple FFT  */
t* FFT_simple(t* x, long n) {
	t* X2 = (t*) malloc(sizeof(struct complex_t) * n);
	t * d, * e, * D, * E;
	long k;

	if (n == 1) {
		X2[0] = x[0];
		return X2;
	}

	e = (t*) malloc(sizeof(struct complex_t) * n/2);
	d = (t*) malloc(sizeof(struct complex_t) * n/2);
	for(k = 0; k < n/2; k++) {
		e[k] = x[2*k];
		d[k] = x[2*k + 1];
	}

	E = FFT_simple(e, n/2);
	D = FFT_simple(d, n/2);

	for(k = 0; k < n/2; k++) {
		/* Multiply entries of D by the twiddle factors e^(-2*pi*i/N * k) */
		D[k] = complex_mult(wnn[k*N/n], D[k]);
	}

	for(k = 0; k < n/2; k++) {
		X2[k]       = complex_add(E[k], D[k]);
		X2[k + n/2] = complex_sub(E[k], D[k]);
	}

	free(D);
	free(E);
	countE++;
	return X2;
}


int main(int argc, char*argv[] )
{
	/*set up memory on the host*/
	cuFloatComplex *h_in = (cuFloatComplex*) malloc(sizeof(cuFloatComplex) * N);
	cuFloatComplex *hFFT_in = (cuFloatComplex*) malloc(sizeof(cuFloatComplex) * N) ;
	cuFloatComplex *hDFT_out = (cuFloatComplex*) malloc(sizeof(cuFloatComplex) * N);
	cuFloatComplex *hFFT_out = (cuFloatComplex*) malloc(sizeof(cuFloatComplex) * N);
	cuFloatComplex *hw_out = (cuFloatComplex*) malloc(sizeof(cuFloatComplex) * N);
	
	long level=0;//the total level of FFT
	long Bsize=1;//The distance between 2 point of butterful
	long NeachL=2;
	long threadN=512;


	/*get th total level of FFT*/
	level=log((float)N)/log((float)2);



	printf("N : %d\n",N);
	

	/*Set up the input data*/
	for(long i=0; i<N; i++){
		h_in[i] = make_cuFloatComplex(sin(i),0);
		hFFT_in[i] = h_in[i];
		hDFT_out[i]=make_cuFloatComplex(0,0);
		hFFT_out[i]=make_cuFloatComplex(0,0);
	}
	


	/*rada algrithm*/
	long i,j,k;    
	cuFloatComplex temp;  
	j = 0;  
  	for(i = 0; i < N -1; i ++){  
        	if(i < j)  
        	{  
            		temp = hFFT_in[i];  
            		hFFT_in[i] = hFFT_in[j];  
            		hFFT_in[j] = temp;  
		}  
        	k = N >> 1;  
  		while( k <= j){  
			j = j - k;  
            		k >>= 1;  
        	}  
        	j = j + k;  
	} 
 


	/*set up memory on the device*/
	cuFloatComplex *d_in, *dDFT_out, *dFFT_in, *dFFT_out, *dw_out;
	cudaMalloc((void**) &d_in, N*sizeof(cuFloatComplex));
	cudaMalloc((void**) &dDFT_out, N*sizeof(cuFloatComplex));
	cudaMalloc((void**) &dFFT_in, N*sizeof(cuFloatComplex));
	cudaMalloc((void**) &dFFT_out, N*sizeof(cuFloatComplex));
	cudaMalloc((void**) &dw_out, N*sizeof(cuFloatComplex));

	
	/*transfer to device*/
	cudaMemcpy(d_in, h_in, N*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(dFFT_in, hFFT_in, N*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
//clock_t start = clock();
struct timeval begin_FFT, end_FFT;
gettimeofday(&begin_FFT, NULL);
	if(N/2<threadN){
		W<<<1, N/2>>>(dw_out);
	}
	else
	{	
		W<<<N/threadN, threadN>>>(dw_out);
	}


	/*launch FFT Kernel*/
	for(long stage=0; stage<level;stage++){
		
		if(N/Bsize<threadN){
			FFT<<<1, N/Bsize>>>(dFFT_out, dFFT_in, stage, Bsize, NeachL,dw_out);
		}
		else{
			FFT<<< N/(Bsize*threadN), threadN>>>(dFFT_out, dFFT_in, stage, Bsize, NeachL,dw_out);
		}		
		Bsize*=2;
		NeachL*=2;
	}
	
	gettimeofday(&end_FFT, NULL);

	fprintf(stdout, "time taken for executing parallel FFT = %lf\n", (end_FFT.tv_sec-begin_FFT.tv_sec) + (end_FFT.tv_usec-begin_FFT.tv_usec)*1.0/1000000);
//clock_t end = clock();
//float secs = (float)(end-start) / CLOCKS_PER_SEC;
//printf("time :%f\n",secs);

	/*transfer to host*/
	cudaMemcpy(hDFT_out, dDFT_out, N*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(hFFT_out, dFFT_out, N*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(hw_out, dw_out, N*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

	
	/* Input for C FFT*/

        t *y;
        t *Y1,*Y2;
        struct timeval begin_SFFT, end_SFFT, begin_DFT, end_DFT;
        y = (t *)malloc(N*sizeof(t));
        Y1 = (t *)malloc(N*sizeof(t));
        Y2 = (t *)malloc(N*sizeof(t));
        long s;
        long m = N;


	for(long i=0; i<N/2;i++){
		wnn[i].re=cuCrealf(hw_out[i]);
		wnn[i].im=cuCimagf(hw_out[i]);
	}

        for(s=0;s<N;s++)
        {
        	y[s].re = cuCrealf(h_in[s]);
        	y[s].im = cuCimagf(h_in[s]);
     }
	gettimeofday(&begin_SFFT, NULL);
        Y1 = FFT_simple(y,m);
       gettimeofday(&end_SFFT, NULL);
        fprintf(stdout, "time taken for executing serial FFT = %lf\n", (end_SFFT.tv_sec-begin_SFFT.tv_sec) + (end_SFFT.tv_usec-begin_SFFT.tv_usec)*1.0/1000000);



/*
    gettimeofday(&begin_DFT, NULL);
    Y2 = DFT(y);
    gettimeofday(&end_DFT, NULL);

    fprintf(stdout, "time taken for executing serial DFT = %lf\n", (end_DFT.tv_sec-begin_DFT.tv_sec) + (end_DFT.tv_usec-begin_DFT.tv_usec)*1.0/1000000);

*/
	/*Test the serial FFT and parallel FFT*/
	double count=0;
	double RMSE=0, temp2, temp3, temp4;
	for(long i=0; i<N; i++){
    temp2 = sqrt((Y1[i].re*Y1[i].re)+(Y1[i].im*Y1[i].im));
    temp3 = sqrt((cuCrealf(hFFT_out[i])*cuCrealf(hFFT_out[i]))+ (cuCimagf(hFFT_out[i])*cuCimagf(hFFT_out[i])));
    temp4 = (temp2-temp3)*(temp2-temp3);
    count = count+temp4;
	}
/*
    //test serial DFT and serial FFT
double count2=0;
double RMSE2=0, temp5, temp6, temp7;
for(long i=0; i<N; i++){
temp5 = sqrt((Y1[i].re*Y1[i].re)+(Y1[i].im*Y1[i].im));
temp6 = sqrt((Y2[i].re*Y2[i].re)+(Y2[i].im*Y2[i].im));
temp7 = (temp5-temp6)*(temp5-temp6);
count2 = count2+temp7;
}
*/
	RMSE=sqrt(count/N);
	printf("RMSE value FFT vs FFT parallel = %f\n",RMSE);
	if(RMSE<=accuracy ){printf("correct\n");}
/*
   RMSE2=sqrt(count2/N);
    printf("RMSE2 DFT vs FFT value = %f\n",RMSE2);
    if(RMSE2<=accuracy ){printf("correct\n");}
*/

	cudaFree(d_in);
	cudaFree(dFFT_in);
	cudaFree(dFFT_out);
	cudaFree(dw_out);
    	free(h_in);
	free(hFFT_in);
	free(hFFT_out);
	free(hw_out);
	free(Y1);
	return EXIT_SUCCESS;
}

