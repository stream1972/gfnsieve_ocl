#pragma warning(disable : 4996)

#define PROG_NAME			"GFNSvCUDA+"
#ifdef DEVICE_CUDA
#define PROG_VERSION		"0.7"
#define PROG_COPY_RIGHT		"2015 Anand Nair (anand.s.nair AT gmail)"
#else
#define PROG_VERSION		"0.7.1"
#define PROG_COPY_RIGHT		"2015 Anand Nair (anand.s.nair AT gmail)"  "\n" \
				"OpenCL port by Roman Trunov (stream AT proxyma ru)"
#endif
#define PROG_TITLE 			PROG_NAME " v" PROG_VERSION " (c) " PROG_COPY_RIGHT

#define __STDC_FORMAT_MACROS

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#if defined(_MSC_VER) && _MSC_VER <= 1600  /* At least MSVC 2010 does not have nor this nor "inttypes.h" */
#define PRIu64 "I64u"
#else
#include <inttypes.h>
#endif
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

#if defined(DEVICE_CUDA) + defined(DEVICE_OPENCL) + defined(DEVICE_SIMULATION) != 1
#error One DEVICE_xxx hardware platform must be defined
#endif

#ifdef DEVICE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef DEVICE_SIMULATION
typedef enum
{
	cudaSuccess,
	cudaErrorMissingConfiguration,
	cudaErrorMemoryAllocation
} cudaError_t;
#define cudaGetErrorString(n) "Fake CUDA error"

/* For simulation, all memory is allocated on PC using plain malloc/free */
static void cudaFreeHost(void *p) { if (p) free(p); }
#define cudaFree cudaFreeHost
static cudaError_t cudaMalloc(void **pHost, size_t size)
{
	*pHost = malloc(size);
	return *pHost ? cudaSuccess : cudaErrorMemoryAllocation;
}
#define cudaMalloc_ro cudaMalloc
#define cudaMalloc_rw cudaMalloc
#define cudaHostAlloc(p, n, f) cudaMalloc(p, n)

static cudaError_t cudaMemcpyWrapper(void *dst, const void *src, size_t count)
{
	memcpy(dst, src, count);
	return cudaSuccess;
}
#define cudaMemcpy_htd(dst, src, count, kind) cudaMemcpyWrapper(dst, src, count)
#define cudaMemcpy_hth(dst, src, count, kind) cudaMemcpyWrapper(dst, src, count)
#endif // DEVICE_SIMULATION

#define PETA 1000000000000000

#define RESULT_BUFFER_SIZE 10000
#define RESULT_BUFFER_COUNT (RESULT_BUFFER_SIZE/2-1)

#if CUDART_VERSION >= 4000
#define SYNC_CALL cudaDeviceSynchronize
#define BLOCKING_SYNC cudaDeviceScheduleBlockingSync
#else
#define SYNC_CALL cudaThreadSynchronize
#define BLOCKING_SYNC cudaDeviceBlockingSync
#endif


typedef uint32_t U32;
typedef uint64_t U64;
typedef U64 u64;
typedef U32 u32;

typedef U32 HALF[3];
typedef U32 FULL[6];

#ifdef PLATFORM_WIN32

#include <intrin.h>

#define MY_TIME clock

#endif



#ifdef PLATFORM_LINUX

#define MY_TIME my_custom_clock

U64 __emulu(U64 a, U64 b)
{
	return a*b;
}

#endif

#ifdef DEVICE_OPENCL
#include "gsv_ocl.h"
#define cudaError_t        cl_int
#define cudaSuccess        CL_SUCCESS
#define cudaGetErrorString ocl_strerror

static cl_int cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
	(void) flags;
	*pHost = malloc(size);
	return *pHost ? CL_SUCCESS : CL_OUT_OF_HOST_MEMORY;
}
static void cudaFreeHost(void *p) { if (p) free(p); }
static cl_int oclCreateBufferGeneric(cl_mem *pmem, size_t size, unsigned flags, ocl_context_t *cont)
{
	cl_int status;

	*pmem = clCreateBuffer(cont->clcontext, flags, size, NULL, &status);
	ocl_diagnose(status, "clCreateBuffer", cont);
	return status;
}
#define cudaMalloc_ro(pp, size) oclCreateBufferGeneric((cl_mem *)(pp), size, CL_MEM_READ_ONLY,  gd.device_ctx)
#define cudaMalloc_rw(pp, size) oclCreateBufferGeneric((cl_mem *)(pp), size, CL_MEM_READ_WRITE, gd.device_ctx)
#define cudaFree(p)             ocl_diagnose( clReleaseMemObject(p), "clReleaseMemObject", gd.device_ctx )

static cl_int hostMemcpyWrapper(void *dest, const void *src, size_t size)
{
	memcpy(dest, src, size);
	return CL_SUCCESS;
}

static cl_int copyToDeviceWrapper(cl_mem dest, void *src, size_t size, ocl_context_t *cont)
{
	cl_int status;

	status = clEnqueueWriteBuffer(cont->cmdQueue, dest, CL_TRUE, 0, size, src, NULL, NULL, NULL);
	return ocl_diagnose(status, "clEnqueueWriteBuffer", cont);
}
#define cudaMemcpy_hth(dest, src, size, flags) hostMemcpyWrapper(dest, src, size)
#define cudaMemcpy_htd(dest, src, size, flags) copyToDeviceWrapper(dest, src, size, gd.device_ctx)
#endif // DEVICE_OPENCL

// Global data structure. Avoid namespace conflict with local variables.
struct global_data {
	U32 n;	// 18..24. might change later after safety analysis
	U32 N;  // 2^n
	U32 N1; // 2^(n+1) -- factors are of the form k.N1+1
	U64 k;  // k from above. Limited to 2^51
	double inv_k; // 1.0/k
	double inv_N1; // 1.0/N1
	double inv_f; // 1.0/k * 1.0/N1
	double inv_f_scale;
	U64 stat;
	U64 factorcount;
	HALF the_f, the_f_inverse;
	U32 the_f_bits;
#if defined(DEVICE_CUDA) || defined(DEVICE_SIMULATION)
	u64 *h_Factor_Mult_Ratio, *d_Factor_Mult_Ratio, *b_Factor_Mult_Ratio;
	u32 *h_Factor_Mult_Ratio1, *d_Factor_Mult_Ratio1, *b_Factor_Mult_Ratio1;
	u64 *h_Init  , *d_Init;
	u32 *h_Init1 , *d_Init1;
	u32 *h_Result, *d_Result;
#endif
#ifdef DEVICE_OPENCL
	/* Same as CUDA but device memory must have cl_mem type */
	u64 *h_Factor_Mult_Ratio,  *b_Factor_Mult_Ratio;  cl_mem d_Factor_Mult_Ratio;
	u32 *h_Factor_Mult_Ratio1, *b_Factor_Mult_Ratio1; cl_mem d_Factor_Mult_Ratio1;
	u64 *h_Init;   cl_mem d_Init;
	u32 *h_Init1;  cl_mem d_Init1;
	u32 *h_Result; cl_mem d_Result;
#endif
	clock_t starttime;
	U32 startp_in_peta;
	U32 endp_in_peta;
	U32 bmax;
	U32 candsInBuffer;
	U32 factorsInBuffer;
	char ckpt[80];
	char fact[80];
	int device_number;
#ifdef DEVICE_CUDA
	cudaDeviceProp device_info;
#endif
#ifdef DEVICE_OPENCL
	ocl_context_t *device_ctx;
	int use_nvidia_workaround;
#endif
	U32 b_blocks_per_grid, blocks_per_grid;
} gd;

// Jacobi symbol
static
int jacobi(U32 m, U32 n)
{
	int c = 1;
	while((m & 1) == 0) { m >>= 1; c = -c; }

	c &= (int) ((n ^ (n>>1) ^ 2) & 2) - 1;

	if(m == 1) {
		return c;
	}

	// quadratic reciprocity
	c *= (int) (( n & m & 2 ) ^ 2) - 1;
	return c * jacobi(n%m, m);
}

//******** S I E V I N G *********
#define SP_COUNT 82000

struct sieve_prime_t {
	U32 p;
	U32 x; // index into array
} sp[SP_COUNT];

#define MAX_PRIME 1024*1024
#define MAX_PRIME_SQRT 1024

#define bit_set(arr, ndx) arr[ndx >> 5] |= (1 << (ndx & 0x1F))
#define bit_test(arr, ndx) (arr[ndx >> 5] & (1 << (ndx & 0x1F)))
#define bit_clear(arr, ndx) arr[ndx >> 5] &= ~(1 << (ndx & 0x1F))

static
U32 invN1(U32 p)
{
	U32 h = 1;

	for(U32 i=0; i <= gd.n; i++)
	{
		if(h & 1) h += p;
		h >>= 1;   // h = h/2 (mod p)
	}
	return h;
}

static
void init_sieve_primes(U64 startk)
{
	// calculate primes

	U32 *sieve = (U32 *) malloc(MAX_PRIME/2);
	memset(sieve, 0xFF, MAX_PRIME/2);

	U32 count = 0, p, bit;

	for(p=3, bit=p>>1; p < MAX_PRIME_SQRT; p += 2, bit++)
	{
		if( bit_test(sieve, bit) )
		{
			sp[count++].p = p;

			for(U32 j=(p*p)>>1; j<MAX_PRIME/2; j += p)
			{
				bit_clear(sieve, j);
			}
		}
	}

	for( ; count < SP_COUNT; p += 2, bit++)
	{
		if( bit_test(sieve, bit) )
		{
			sp[count++].p = p;
		}
	}

	free(sieve);

	// compute x
	for(U32 i=0; i<SP_COUNT; i++)
	{
		p = sp[i].p;

		U32 modp = startk%p;
		U32 x = invN1(p) + modp;
		if(x >= p) x -= p;
		if(x) x = p-x;

		sp[i].x = x;
	}
}

#define SIEVE_SIZE 16384
#define SIEVE_BITS (SIEVE_SIZE*sizeof(U32)*8)

U32 sieve[SIEVE_SIZE];

static
void sieve_iteration()
{
	memset(sieve, 0xFF, sizeof(sieve));

	for(U32 i=0; i < SP_COUNT; i++)
	{
		U32 p = sp[i].p, x = sp[i].x;
		for(; x < SIEVE_BITS; x += p)
		{
			bit_clear(sieve, x);
		}
		sp[i].x = x - SIEVE_BITS;
	}
}

//******** END S I E V I N G END *********

#if !defined PLATFORM_WIN32 && !defined PLATFORM_LINUX
#error No platforms defined
#endif

#define M_2_POW_32 4294967296.0

#define MASK_24 0xffffff

static void modFULLslow(const FULL a, HALF b);
static void expmodHALF(U32 b, HALF ret);
static void expmodHALFcheck(const HALF init, HALF ret);
static void calc_inverse(void);
static void calc_ratio(const HALF num, const HALF den, HALF ratio);

#ifdef PLATFORM_WIN32
#include <Windows.h>

volatile int term_requested = 0;

static BOOL WINAPI CtrlHandler( DWORD fdwCtrlType )
{
	(void) fdwCtrlType;
	term_requested = 1;
	printf("\nTermination requested\n");
	return TRUE;
}
#endif // PLATFORM_WIN32

#ifdef PLATFORM_LINUX
#include <signal.h>

clock_t my_custom_clock()
{
	timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	double tt = (double) t.tv_sec + (double) t.tv_nsec * 1.0e-9;

	return (clock_t) (tt * CLOCKS_PER_SEC);
}

volatile sig_atomic_t term_requested = 0;

void my_sig_handler(int sig)
{
	term_requested = 1;
	printf("\nTermination requested\n");
	signal(sig, my_sig_handler);
}
#endif // PLATFORM_LINUX

static
U64 cvt_q(const HALF h)
{
	return ((U64) h[1] << 32) + h[0];
}

static
U64 cvt_q_hi(const HALF h)
{
	return ((U64) h[2] << 32) + h[1];
}

static
double cvt_dbl(const HALF h)
{
	return ((double) h[2] * M_2_POW_32 + (double) h[1]) * M_2_POW_32 + (double) h[0];
}

static
void initHALF2(HALF a, U64 q, U32 d)
{
	a[0] = (U32) q;
	a[1] = (U32) (q >> 32);
	a[2] = d;
}

static
void initHALF(HALF a, U64 q)
{
	initHALF2(a, q, 0);
}

static
void copyHALF(HALF dest, const HALF src)
{
	dest[0] = src[0], dest[1] = src[1], dest[2] = src[2];
}

static
void divmodHALFsingle(HALF a, U32 b, U32 &m) // a /= b
{
	U64 q;
	U64 r = 0;

	for(int i=2; i >= 0; i--)
	{
		r = (r << 32) + a[i];
		q = r/b; a[i] = (U32) q;
		r -= q * b;
	}
	m = (U32) r;
}

#if 0 // Not used
static
void divHALFsingle(HALF a, U32 b) // a /= b
{
	U32 dummy;

	divmodHALFsingle(a,b,dummy);
}
#endif

static
void shrHALF(HALF a, U32 b) // a >>= b
{
	while(b >= 32)
	{
		a[0] = a[1], a[1] = a[2], a[2] = 0;
		b -= 32;
	}
	if(b > 0)
	{
		a[0] = (a[0] >> b) | (a[1] << (32-b));
		a[1] = (a[1] >> b) | (a[2] << (32-b));
		a[2] >>= b;
	}
}

static
void shlHALF(HALF a, U32 b) // a <<= b
{
	while(b >= 32)
	{
		a[2] = a[1], a[1] = a[0], a[0] = 0;
		b -= 32;
	}
	if(b > 0)
	{
		a[2] = (a[2] << b) | (a[1] >> (32-b));
		a[1] = (a[1] << b) | (a[0] >> (32-b));
		a[0] <<= b;
	}
}

#if 0 // Not used
static
void hiHALF(const FULL f, HALF h)
{
	h[0] = f[3];
	h[1] = f[4];
	h[2] = f[5];
}
#endif

static
void loHALF(const FULL f, HALF h)
{
	h[0] = f[0];
	h[1] = f[1];
	h[2] = f[2];
}

static
void mulHALF(const HALF a, const HALF b, FULL c)
{
	U64 t;
	U32 m;
	HALF w;

	m = b[0];

	t  = __emulu(m, a[0]); c[0] = (U32) t; t >>= 32;
	t += __emulu(m, a[1]); w[0] = (U32) t; t >>= 32;
	t += __emulu(m, a[2]); w[1] = (U32) t; w[2] = (U32) (t >> 32);

	m = b[1];

	t  = __emulu(m, a[0]) + w[0]; c[1] = (U32) t; t >>= 32;
	t += __emulu(m, a[1]) + w[1]; w[0] = (U32) t; t >>= 32;
	t += __emulu(m, a[2]) + w[2]; w[1] = (U32) t; w[2] = (U32) (t >> 32);

	m = b[2];

	t  = __emulu(m, a[0]) + w[0]; c[2] = (U32) t; t >>= 32;
	t += __emulu(m, a[1]) + w[1]; c[3] = (U32) t; t >>= 32;
	t += __emulu(m, a[2]) + w[2]; c[4] = (U32) t; c[5] = (U32) (t >> 32);
}

static
void mulHALFpartial(const HALF a, const HALF b, HALF c)
{
	U64 t;
	U32 m;
	HALF w;

	m = b[0];

	t  = __emulu(m, a[0]); c[0] = (U32) t; t >>= 32;
	t += __emulu(m, a[1]); w[0] = (U32) t; t >>= 32;
	t += __emulu(m, a[2]); w[1] = (U32) t;

	m = b[1];

	t  = __emulu(m, a[0]) + w[0]; c[1] = (U32) t; t >>= 32;
	t += __emulu(m, a[1]) + w[1];

	m = b[2];

	t  += __emulu(m, a[0]); c[2] = (U32) t;
}

static
void mulHALFpartialhi(const HALF a, const HALF b, HALF c)
{
	U64 t;
	U32 m;
	HALF w;

	m = b[0];

	t  = __emulu(m, a[0]); t >>= 32;
	t += __emulu(m, a[1]); w[0] = (U32) t; t >>= 32;
	t += __emulu(m, a[2]); w[1] = (U32) t; w[2] = (U32) (t >> 32);

	m = b[1];

	t  = __emulu(m, a[0]) + w[0]; t >>= 32;
	t += __emulu(m, a[1]) + w[1]; w[0] = (U32) t; t >>= 32;
	t += __emulu(m, a[2]) + w[2]; w[1] = (U32) t; w[2] = (U32) (t >> 32);

	m = b[2];

	if( m )
	{
		t  = __emulu(m, a[0]) + w[0]; t >>= 32;
		t += __emulu(m, a[1]) + w[1]; c[0] = (U32) t; t >>= 32;
		t += __emulu(m, a[2]) + w[2]; c[1] = (U32) t; c[2] = (U32) (t >> 32);
	}
	else
	{
		c[0] = w[1];
		c[1] = w[2];
		c[2] = 0;
	}
}

static
void squareHALF(const HALF a, FULL b)
{
	U64 t;
	U32 m;
	U32 w1, w2, w3, w4;

	// Calculate intermediate products.
	m = a[0];
	t  = __emulu(m, a[1]); w1 = (U32) t; t >>= 32;
	t += __emulu(m, a[2]); w2 = (U32) t; t >>= 32;

	m = a[1];
	t += __emulu(m, a[2]); w3 = (U32) t; w4 = (U32) (t >> 32);

	// Final = square products + 2*intermediate products
	m = a[0]; t = __emulu(m, m); b[0] = (U32) t; b[1] = (t >> 32);
	m = a[1]; t = __emulu(m, m); b[2] = (U32) t; b[3] = (t >> 32);
	m = a[2]; t = __emulu(m, m); b[4] = (U32) t; b[5] = (t >> 32);

	t  = __emulu(w1, 2) + b[1]; b[1] = (U32) t; t >>= 32;
	t += __emulu(w2, 2) + b[2]; b[2] = (U32) t; t >>= 32;
	t += __emulu(w3, 2) + b[3]; b[3] = (U32) t; t >>= 32;
	t += __emulu(w4, 2) + b[4]; b[4] = (U32) t; t >>= 32;
	t += b[5]; b[5] = (U32) t;
}

static
int cmpHALF(const HALF a, const HALF b)
{
	if(a[2] > b[2]) return 1;
	if(a[2] < b[2]) return -1;
	if(a[1] > b[1]) return 1;
	if(a[1] < b[1]) return -1;
	if(a[0] > b[0]) return 1;
	if(a[0] < b[0]) return -1;

	return 0;
}

static
void addHALF(HALF a, const HALF b) // a += b
{
	U64 t;

	t  = (U64) a[0] + b[0]; a[0] = (U32) t; t >>= 32;
	t += (U64) a[1] + b[1]; a[1] = (U32) t; t >>= 32;
	t += (U64) a[2] + b[2]; a[2] = (U32) t;
}

static
void incHALF(HALF a) // a++
{
	if( ++a[0] == 0)
	{
		if( ++a[1] == 0)
		{
			++a[2];
		}
	}
}

static
int subHALF(HALF a, const HALF b)
{
	U64 t;

	t = (U64) a[0] - (U64) b[0];     a[0] = (U32) t; t = (t >> 32) & 1;
	t = (U64) a[1] - (U64) b[1] - t; a[1] = (U32) t; t = (t >> 32) & 1;
	t = (U64) a[2] - (U64) b[2] - t; a[2] = (U32) t; t = (t >> 32) & 1;
	return (U32) t;
}

static
void mul_64_32(HALF a, U64 q, U32 b)
{
	U32 q0 = (U32) q, q1 = (U32) (q >> 32);

	U64 t;

	t  = __emulu(b, q0); a[0] = (U32) t; t >>= 32;

	t += __emulu(b, q1); a[1] = (U32) t; a[2] = (U32) (t >> 32);
}

static
char *HALF2Str(const HALF a)
{
	U64 q;
	U32 r;
	HALF tmp;

	copyHALF(tmp, a);

	divmodHALFsingle(tmp, 1000000000, r);

	q = cvt_q(tmp);

	static char buf[32] = "";

	sprintf(buf, "%" PRIu64 "%09u", q, r);

	return buf;
}

// result must fit within a FULL. no check performed.
static
void mulFULLsingle(const FULL a, const U32 b, FULL c) // c = a * b;
{
	U64 t;

	t  = __emulu(b, a[0]); c[0] = (U32) t; t >>= 32;
	t += __emulu(b, a[1]); c[1] = (U32) t; t >>= 32;
	t += __emulu(b, a[2]); c[2] = (U32) t; t >>= 32;
	t += __emulu(b, a[3]); c[3] = (U32) t; t >>= 32;
	t += __emulu(b, a[4]); c[4] = (U32) t; t >>= 32;
	t += __emulu(b, a[5]); c[5] = (U32) t;
}

static
int subFULL(const FULL a, const FULL b, FULL c) // c= a-b. borrow out is returned
{
	U64 t;

	t = (U64) a[0] - (U64) b[0];     c[0] = (U32) t; t = (t >> 32) & 1;
	t = (U64) a[1] - (U64) b[1] - t; c[1] = (U32) t; t = (t >> 32) & 1;
	t = (U64) a[2] - (U64) b[2] - t; c[2] = (U32) t; t = (t >> 32) & 1;
	t = (U64) a[3] - (U64) b[3] - t; c[3] = (U32) t; t = (t >> 32) & 1;
	t = (U64) a[4] - (U64) b[4] - t; c[4] = (U32) t; t = (t >> 32) & 1;
	t = (U64) a[5] - (U64) b[5] - t; c[5] = (U32) t; t = (t >> 32) & 1;

	return (int) (U32) t;
}

static
U32 msb(U32 x)
{
	U32 b = 0;
	if (x & 0xFFFF0000) { b |= 16; x >>= 16; }
	if (x & 0xFF00)     { b |= 8; x >>= 8; }
	if (x & 0xF0)       { b |= 4; x >>= 4; }
	if (x & 0xC)        { b |= 2; x >>= 2; }
	if (x & 0x2)        { b |= 1; }

	return b;
}

static
U32 bitsHALF(const HALF a)
{
	return 1 + (a[2] ? (msb(a[2])+64) : (msb(a[1])+32));
}

// process a single k. k is available in gd.k
static
void processk(U32 cand_per_fac)
{
	initHALF(gd.the_f, gd.k); shlHALF(gd.the_f, gd.n+1); gd.the_f[0]++;

	U32 p;
	int qnr = 1;

	//   find QNR
	for(U32 i=0; i < 128; i++) {
		p = sp[i].p;
		U32 modp = (((gd.k % p) << (gd.n+1)) + 1) % p;
		if (jacobi(modp, p) == -1) {
			qnr = -1;
			break;
		}
	}

	if(qnr != -1) {
		printf("\nUnable to find qnr for %s\n", HALF2Str(gd.the_f));
		return;
	}

	//   calculate init (p^k) and multiplier (init^2)
	calc_inverse();

	HALF init, check;

	expmodHALF(p, init);
	expmodHALFcheck(init, check);
	check[0]++;
	if(cmpHALF(check, gd.the_f) != 0) // not a prime
		return;

	gd.h_Init[gd.candsInBuffer] = cvt_q(init);
	gd.h_Init1[gd.candsInBuffer] = init[2];
	gd.candsInBuffer++;

	FULL finit2; HALF init2; squareHALF(init, finit2); modFULLslow(finit2, init2);
	HALF ratio; calc_ratio(init2, gd.the_f, ratio);

	// Previous versions always executed this loop.
	// Now, with dynamic iteration count, this might not be executed.
 	if (cand_per_fac > 1)
	{
		U32 cands_left = cand_per_fac-1;

		HALF tmp; copyHALF(tmp, init);
		HALF q, r, s;

		do {
			mulHALFpartialhi(ratio, init, q);
			incHALF(q);
			mulHALFpartial(init, init2, r);
			mulHALFpartial(gd.the_f, q, s);
			if( subHALF(r, s) )
			{
				addHALF(r, gd.the_f);
			}
			copyHALF(init, r);
			gd.h_Init[gd.candsInBuffer] = cvt_q(init);
			gd.h_Init1[gd.candsInBuffer] = init[2];
			gd.candsInBuffer++;

			cands_left--;
		} while (cands_left);

		mulHALF(tmp, init, finit2); modFULLslow(finit2, init2);
		calc_ratio(init2, gd.the_f, ratio);
	}

	gd.h_Factor_Mult_Ratio[gd.factorsInBuffer*3  ] = cvt_q(gd.the_f);  gd.h_Factor_Mult_Ratio1[gd.factorsInBuffer*3  ] = gd.the_f[2];
	gd.h_Factor_Mult_Ratio[gd.factorsInBuffer*3+1] = cvt_q(init2);     gd.h_Factor_Mult_Ratio1[gd.factorsInBuffer*3+1] = init2[2];
	gd.h_Factor_Mult_Ratio[gd.factorsInBuffer*3+2] = cvt_q_hi(ratio);  gd.h_Factor_Mult_Ratio1[gd.factorsInBuffer*3+2] = ratio[0];
	gd.factorsInBuffer++;
}

static
int Str2HALF(const char *s, HALF a, U32 * peta)
{
	int len = strlen(s);

	U32 hi_peta = 0;
	U64 lo_peta = 0;

	if(len > 24)
		return 1;

	for(int i=0; i < len; i++) {
		if(s[i] < '0' || s[i] > '9')
			return 1;

		if(i < (len-15)) {
			hi_peta = hi_peta * 10 + (s[i]-'0');
		}
		else {
			lo_peta = lo_peta * 10 + (s[i]-'0');
		}
	}


	mul_64_32(a, PETA, hi_peta);
	HALF tmp; initHALF(tmp, lo_peta);
	addHALF(a, tmp);

	*peta = hi_peta;

	return 0;
}

static
U64 peta_to_k(U32 p_in_peta)
{
	HALF tmp;
	mul_64_32(tmp, PETA, p_in_peta); shrHALF(tmp, gd.n+1);

	return cvt_q(tmp);
}

static
void copyFULL(FULL dest, const FULL src)
{
	for(int i=0; i < 6; i++)
		dest[i] = src[i];
}

static
void shrFULL(FULL a, U32 b) // a >>= b
{
	while(b >= 32)
	{
		for(int i=0; i < 5; i++)
			a[i] = a[i+1];
		a[5] = 0;
		b -= 32;
	}
	if(b > 0)
	{
		for(int i=0; i < 5; i++)
			a[i] = (a[i] >> b) | (a[i+1] << (32-b));
		a[5] >>= b;
	}
}

static
void modFULLslow(const FULL a, HALF b) // b = a mod k.N1+1
{
	HALF a_hi, quot;
	FULL tmp;

	copyFULL(tmp, a);
	shrFULL(tmp, gd.the_f_bits-1);
	loHALF(tmp, a_hi);

	mulHALF(a_hi, gd.the_f_inverse, tmp);
	shrFULL(tmp, gd.the_f_bits+1);
	loHALF(tmp, quot);

	mulHALF(quot, gd.the_f, tmp);

	subFULL(a, tmp, tmp);
	loHALF(tmp, b);
	while(cmpHALF(b, gd.the_f) >= 0)
		subHALF(b, gd.the_f);
}

static
void expmodHALF(U32 b, HALF ret) // compute b^k mod (k*N1+1)
{
	U64 q = 0x8000000000000000;

	while ((q & gd.k) == 0) q >>= 1;

	HALF e;
	FULL f;

	initHALF(e, b);

	q >>= 1;
	while(q)
	{
		squareHALF(e, f);
		if(q & gd.k)
		{
			mulFULLsingle(f, b, f);
		}
		modFULLslow(f, e);
		q >>= 1;
	}

	copyHALF( ret, e );
}

static
void expmodHALFcheck(const HALF init, HALF ret) // compute init^N mod (k*N1+1)
{
	HALF e;
	FULL f;

	copyHALF(e,  init);

	for(U32 i=0; i<gd.n; i++)
	{
		squareHALF(e, f);
		modFULLslow(f, e);
	}

	copyHALF( ret, e );
}


static
void calc_inverse()
{
	gd.the_f_bits = bitsHALF(gd.the_f);

	HALF num1; initHALF(num1, 1); shlHALF(num1, gd.the_f_bits);

	initHALF(gd.the_f_inverse, 0);

	for(U32 i=0; i<gd.the_f_bits+1; i++)
	{
		// Double num
		shlHALF(gd.the_f_inverse, 1);

		if(cmpHALF(num1, gd.the_f) >= 0)
		{
			subHALF(num1, gd.the_f);
			gd.the_f_inverse[0] |= 1;
		}

		shlHALF(num1, 1);
	}
}

static
void calc_ratio(const HALF num, const HALF den, HALF ratio)
{
	HALF num1; copyHALF(num1, num);

	initHALF(ratio, 0);

	for(int i=0; i<96; i++)
	{
		// Double num
		shlHALF(num1, 1);

		shlHALF(ratio, 1);

		if(cmpHALF(num1, den) >= 0)
		{
			subHALF(num1, den);
			ratio[0] |= 1;
		}
	}
}


static
U64 read_last_factor()
{
	char buf[80] = "";

	printf("Reading factor file for last factor\n");

	FILE * fp = fopen(gd.fact, "r");
	if(fp == NULL) {
		printf("Unable to read factor file '%s'. Starting the range from beginning\n", gd.fact);
		return 0;
	}

	while(!feof(fp)) {
		fgets(buf, sizeof(buf)-1, fp);
	}
	fclose(fp);


	U64 k = 0; char tmpbuf[80] = ""; HALF f; U32 peta = 0;
	if(strchr(buf, '|') && sscanf(buf, "%s", tmpbuf) == 1 && Str2HALF(tmpbuf, f, &peta) == 0 && f[0]%gd.N1 == 1) {
		printf("Found factor %s\n", tmpbuf);
		shrHALF(f, gd.n+1);
		k = cvt_q(f);
	}
	else {
		k = 0;
	}

	if(k == 0) {
		printf("Unable to extract a factor from factor file. Last read line was:\n'%s'\n", buf);
	}
	return k;
}

static
U64 read_checkpoint(U64 startk, U64 endk)
{
	FILE *fp = fopen(gd.ckpt, "r");
	if (fp == NULL)
		return 0;

	U64 k = 0;

	char buf[80] = "";

	int count = fscanf(fp, "%s", buf);
	fclose(fp);

	if(count < 1) {
		printf("Unable to read checkpoint\n");
		k = read_last_factor();
	}
	else {
		HALF startf; initHALF(startf, startk); shlHALF(startf, gd.n+1); startf[0]++;

		HALF tmp; U32 p_val = 0; Str2HALF(buf, tmp, &p_val);

		// This if statement allows restarting from both factor as well as k value
		if(cmpHALF(tmp,startf) >= 0) {
			shrHALF(tmp, gd.n+1);
		}

		k = cvt_q(tmp);

		if(k >= startk && k <= endk + 4*SIEVE_BITS) {
			// Looks good
		}
		else {
			printf("Warning: Bad k value %" PRIu64 " found in checkpoint. Ignoring it...\n", k);
			k = read_last_factor();
		}
	}
	return k;
}

static
void write_checkpoint(U64 k, bool force = false)
{
#ifndef DEVICE_SIMULATION
	static clock_t next_write_time = 0;

	clock_t curr_time = MY_TIME();

	if (force || (curr_time > next_write_time)) {
		FILE *fp = fopen(gd.ckpt, "w");
		if (fp == NULL) {
			printf("Unable to write to checkpoint file\n");
			exit(1);
		}

		fprintf(fp, "%" PRIu64 "", k);
		fflush(fp);
		fclose(fp);
		next_write_time = curr_time + CLOCKS_PER_SEC * 60; // Write once a minute
	}
#else
	/* Don't write durinf simulation to avoid occasional skip of a range */
	(void) k;
	(void) force;
#endif
}

static
void write_factor(FILE *fp, const HALF factor, U32 base, bool screen_out)
{
	char buf[80];
	sprintf(buf, "%s | %u^%u+1", HALF2Str(factor), base, gd.N);

	fprintf(fp, "%s\n", buf);

	if (screen_out)
		printf("%-78s\n", buf);
}

static
void write_factor_batch(U32 factor_count)
{
	FILE *fp = fopen(gd.fact, "a");
	if (fp == NULL) {
		printf("Unable to write to factor file\n");
		exit(1);
	}

	for(U32 ii=0; ii < factor_count; ii++) {
		HALF factor;
		U32 b_value = gd.h_Result[2*ii+1];
		U32 f_index = gd.h_Result[2*ii+2];
		initHALF2(factor, gd.b_Factor_Mult_Ratio[f_index*3], gd.b_Factor_Mult_Ratio1[f_index*3]);
		write_factor( fp, factor, b_value, ii < 4 ); // Reduced SCREEN output: atmost 4 factors per batch
	}

	fflush(fp);
	fclose(fp);
}

static
void write_factor_header()
{
	FILE *fp = fopen(gd.fact, "a");
	if (fp == NULL) {
		printf("Unable to write to factor file\n");
		exit(1);
	}

	fprintf(fp, "GFN Sieve for k^%u+1 [k == 2 to %u]\n\n", gd.N, gd.bmax);
	fflush(fp);
	fclose(fp);
}

static
void print_status(HALF lastf)
{
	static clock_t next_status_time = 0;
	static U64 last_factor_count = 0;
	static char status_line[80] = "";

	clock_t curr_time = MY_TIME();

	U32 hh, mm, secs;
	double speed, P_day;

	if (curr_time >= next_status_time) {
		speed = (double) (gd.stat * CLOCKS_PER_SEC) / (curr_time-gd.starttime);

		P_day = speed * 86400.0 * log(cvt_dbl(lastf)) * (double) gd.N * 1.0e-15;

		long eta = ((double) gd.endp_in_peta - cvt_dbl(lastf) * 1.0e-15) * 86400.0 / P_day;
		if (eta < 1) eta = 1;

		secs = (U32) eta;
		hh = secs / 3600;
		secs -= hh * 3600;
		mm = secs / 60;
		secs -= mm * 60;

		sprintf(status_line, "%s %.1f/s (%.1fP/day) Found %" PRIu64 " ETA %uh%02um  "
		      , HALF2Str(lastf)
		      , speed
		      , P_day
		      , gd.factorcount
		      , hh
		      , mm);

		printf("%s\r", status_line);
		fflush(stdout);
		gd.starttime = curr_time;
		gd.stat = 0;
		next_status_time = curr_time + CLOCKS_PER_SEC * 3; // Write once every 3 sec
		last_factor_count = gd.factorcount;
	}
	else {
		// Did a factor overwrite our status line?
		if(gd.factorcount > last_factor_count) {
			printf("%s\r", status_line);
			fflush(stdout);
			last_factor_count = gd.factorcount;
		}
	}

}

#ifdef DEVICE_CUDA
// x86 style intrinsics for multi-precision arithmetic

__device__ static unsigned int __add(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm volatile ("add.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}

__device__ static unsigned int __adc(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}

__device__ static unsigned int __sub(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __sbb(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}

#if (__CUDA_ARCH__ < 200)

__device__ static unsigned int __umul24hi(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm volatile ("mul24.hi.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}

#define UMUL24(a,b) (__umul24(a,b) & MASK_24)
#define UMUL24HI(a,b) (__umul24hi(a,b) >> 8)

#endif


#if (__CUDA_ARCH__ >= 200) && (CUDART_VERSION >= 4010)

__device__ static unsigned int __mad(unsigned int a, unsigned int b, unsigned int c)
{
  unsigned int r;
  asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
  return r;
}

__device__ static unsigned int __madhi(unsigned int a, unsigned int b, unsigned int c)
{
  unsigned int r;
  asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
  return r;
}

__device__ static unsigned int __madc(unsigned int a, unsigned int b, unsigned int c)
{
  unsigned int r;
  asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
  return r;
}

__device__ static unsigned int __madchi(unsigned int a, unsigned int b, unsigned int c)
{
  unsigned int r;
  asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
  return r;
}

#endif

__device__ static void process63
(
	u64 f, /* factor to be checked */
	u64 i, /* initial value */
	u64 m, /* multiplier - should be a power of i */
	u64 r, /* inverse of multiplier */
	u32 bmax, /* range of admissible candidate. only hits < bmax need to be considered. bmax should be odd. */
	u32 count, /* how many iterations thru the main loop. inversely proportional to the power of i used in multiplier */
	u32 *RES, /* global array to which successes, if any, should be returned */
	u32 si
)
{
	do { /* loop down */

		/* only bothered about even valued candidates. choose either i or f-i */
		//u64 tmp = (i & 1) - 1; /* tmp = FF..FF or 00..00 */
		//i = (i & tmp) | ((f - i) & ~tmp);
		if(i & 1)
			i = f-i;

		/* check if we have an even-valued success */
		if(i < bmax) {
			u32 count = atomicInc(&RES[0], RESULT_BUFFER_COUNT) + 1;
			RES[count*2-1] = (u32) i; /* TODO: Add array bounds check */
			RES[count*2] = si;
		}

		/* compute next iteration of i in 4 steps */
		u64 q = __umul64hi(i, r) + 1;	// step 1 - compute quotient
		u64 p = i*m;					// step 2 - compute actual product i * m
		u64 g = q*f;					// step 3 - compute approximate quantity to subtract

		// step 4: compute the difference (p - g) with branchless code for correcting negatives
		i = (p-g);
		if (i & 0x8000000000000000)
			i += f;
	} while( --count ); // End of countdown loop
}


#if (__CUDA_ARCH__ < 200) || (CUDART_VERSION < 4010)

//__device__ static void process64
//(
//	u64 f, /* factor to be checked */
//	u64 i, /* initial value */
//	u64 m, /* multiplier - should be a power of i */
//	u64 r, /* inverse of multiplier */
//	u32 bmax, /* range of admissible candidate. only hits < bmax need to be considered. bmax should be odd. */
//	u32 count, /* how many iterations thru the main loop. inversely proportional to the power of i used in multiplier */
//	u32 *RES, /* global array to which successes, if any, should be returned */
//	u32 si
//)
//{
//	do { /* loop down */
//
//		/* only bothered about even valued candidates. choose either i or f-i */
//		u64 tmp = (i & 1) - 1; /* tmp = FF..FF or 00..00 */
//		i = (i & tmp) | ((f - i) & ~tmp);
//
//		/* check if we have an even-valued success */
//		if(i < bmax) {
//			u32 count = atomicInc(RES, RESULT_BUFFER_COUNT) + 1;
//			RES[count*2-1] = (u32) i; /* TODO: Add array bounds check */
//			RES[count*2] = si;
//		}
//
//		/* compute next iteration of i in 4 steps */
//
//		/* step 1 - compute quotient = (HIGH64 of i*r) + 1 */
//		u64 q = __umul64hi(i, r) + 1;
//
//
//		/* step 2 - compute full product i * m */
//		u64 p = i*m;
//		u64 p_hi = __umul64hi(i, m);
//
//
//		/* step 3 - compute qoutient*factor */
//		u64 g = q*f;
//		u64 g_hi = __umul64hi(q, f);
//
//
//		/* STEP 4: i = (p - g) with branchless code for correcting negatives */
//		u64 t = (p_hi - g_hi) - ((p < g) ? 1 : 0); /* t = FF...FF if over-adjusted or 0 if perfect */
//		i = (p - g) + (f & t);
//	} while( --count ); // End of countdown loop
//}

__device__ static void process64
(
	u64 f, /* factor to be checked */
	u64 i, /* initial value */
	u64 m, /* multiplier - should be a power of i */
	u64 r, /* inverse of multiplier */
	u32 bmax, /* range of admissible candidate. only hits < bmax need to be considered. bmax should be odd. */
	u32 count, /* how many iterations thru the main loop. inversely proportional to the power of i used in multiplier */
	u32 *RES, /* global array to which successes, if any, should be returned */
	u32 si
)
{
	u32 i0 = (u32) i, i1 = (u32) (i >> 32);
	u32 f0 = (u32) f, f1 = (u32) (f >> 32);
	u32 m0 = (u32) m, m1 = (u32) (m >> 32);
	u32 r0 = (u32) r, r1 = (u32) (r >> 32);

	do { /* loop down */

		/* only bothered about even valued candidates. choose either i or f-i */
		//u32 tmp = (i0 & 1) - 1; /* tmp = FF..FF or 00..00 */
		//i0 = (i0 & tmp) | (__sub(f0, i0) & ~tmp);
		//i1 = (i1 & tmp) | (__sbb(f1, i1) & ~tmp);
		if(i0 & 1) {
			i0 = __sub(f0, i0);
			i1 = __sbb(f1, i1);
		}

		/* check if we have an even-valued success */
		if( (i1 == 0) && (i0 < bmax) ) {
			u32 count = atomicInc(RES, RESULT_BUFFER_COUNT) + 1;
			RES[count*2-1] = i0; /* TODO: Add array bounds check */
			RES[count*2] = si;
		}

		/* compute next iteration of i in 4 steps */

		/* step 1 - compute quotient = (HIGH64 of i*r) + 1 */
		u32 t, q0, q1;

		t  = __umulhi(i0, r0);

		t  = __add( i0*r1, t ); // replace with mad for sm20 & higher
		q0 = __adc( __umulhi(i0, r1), 1 );

		t  = __add( i1*r0, t );
		q0 = __adc( __umulhi(i1, r0), q0 );
		q1 = __adc( 0, 0 );

		q0 = __add( i1*r1, q0 );
		q1 = __adc( __umulhi(i1, r1), q1 );


		/* step 2 - compute full product i * m */
		u32 p0, p1, p2;

		p0 = i0*m0;
		p1 = __umulhi( i0, m0 );

		p1 = __add( i0*m1, p1 );
		p2 = __adc( __umulhi(i0, m1), 0 );

		p1 = __add( i1*m0, p1 );
		p2 = __adc( __umulhi(i1, m0), p2 );

		p2 += i1*m1;


		/* step 3 - compute qoutient*factor and subtract from P */
		u32 g0, g1, g2;

		g0 = q0*f0;
		g1 = __umulhi( q0, f0 );

		g1 = __add( q0*f1, g1 );
		g2 = __adc( __umulhi(q0, f1), 0 );

		g1 = __add( q1*f0, g1 );
		g2 = __adc( __umulhi(q1, f0), g2 );

		g2 += q1*f1;


		/* STEP 4: i = (p - g) with branchless code for correcting negatives */

		i0 = __sub( p0, g0 );
		i1 = __sbb( p1, g1 );
		p2 = __sbb( p2, g2 );

		if (p2) {
			i0 = __add( i0, f0);
			i1 = __adc( i1, f1);
		}
	} while( --count ); // End of countdown loop
}

__device__ static void process79
(
	u64 f, /* factor to be checked */
	u64 i, /* initial value */
	u64 m, /* multiplier - should be a power of i */
	u64 r, /* inverse of multiplier */
	u32 bmax, /* range of admissible candidate. only hits < bmax need to be considered. bmax should be odd. */
	u32 count, /* how many iterations thru the main loop. inversely proportional to the power of i used in multiplier */
	u32 *RES, /* global array to which successes, if any, should be returned */
	u32 si,
	u32 f_,
	u32 i_,
	u32 m_,
	u32 r_
)
{
	u32 i0 = (u32) i, i1 = (u32) (i >> 32), i2 = i_;
	u32 f0 = (u32) f, f1 = (u32) (f >> 32), f2 = f_;
	u32 m0 = (u32) m, m1 = (u32) (m >> 32), m2 = m_;
	u32 r0 = r_, r1 = (u32) r, r2 = (u32) (r >> 32);

	do { /* loop down */

		/* only bothered about even valued candidates. choose either i or f-i */
		//u32 tmp = (i0 & 1) - 1; /* tmp = FF..FF or 00..00 */
		//i0 = (i0 & tmp) | (__sub(f0, i0) & ~tmp);
		//i1 = (i1 & tmp) | (__sbb(f1, i1) & ~tmp);
		//i2 = (i2 & tmp) | (__sbb(f2, i2) & ~tmp);
		if(i0 & 1) {
			i0 = __sub(f0, i0);
			i1 = __sbb(f1, i1);
			i2 = __sbb(f2, i2);
		}

		/* check if we have an even-valued success */
		if( ((i1|i2) == 0) && (i0 < bmax) ) {
			u32 count = atomicInc(RES, RESULT_BUFFER_COUNT) + 1;
			RES[count*2-1] = i0; /* TODO: Add array bounds check */
			RES[count*2] = si;
		}

		/* compute next iteration of i in 4 steps */

		/* step 1 - compute quotient = (HIGH96 of i*r) + 1 */
		u32 t1, t2, q0, q1, q2;

		//t0 = i0*r0;
		t1 = __umulhi(i0, r0);

		t1 = __add( i0*r1, t1 );
		t2 = __adc( __umulhi(i0, r1), 0 );

		t2 = __add( i0*r2, t2 );
		q0 = __adc( __umulhi(i0, r2), 1 );

		t1 = __add( i1*r0, t1 );
		t2 = __adc( __umulhi(i1, r0), t2 );
		q0 = __adc( i1*r2, q0 );
		q1 = __adc( __umulhi(i1,r2), 0 );

		t2 = __add( i1*r1, t2 );
		q0 = __adc( __umulhi(i1,r1), q0 );
		q1 = __adc( q1, 0 );


		t2 = __add( i2*r0, t2 );
		q0 = __adc( __umulhi(i2,r0), q0 );
		q1 = __adc( i2*r2, q1 );
		q2 = __adc( __umulhi(i2,r2), 0 );

		q0 = __add( i2*r1, q0 );
		q1 = __adc( __umulhi(i2,r1), q1 );
		q2 = __adc( q2, 0 );




		/* step 2 - compute full product i * m */
		u32 p0, p1, p2;

		p0 = i0*m0;
		p1 = __umulhi( i0, m0 );

		p1 = __add( i0*m1, p1 );
		p2 = __adc( __umulhi(i0, m1), 0 );

		p1 = __add( i1*m0, p1 );
		p2 = __adc( __umulhi(i1, m0), p2 );

		p2 += i0*m2 + i1*m1 + i2*m0;


		/* step 3 - compute qoutient*factor and subtract from P */
		u32 g0, g1, g2;

		g0 = q0*f0;
		g1 = __umulhi( q0, f0 );

		g1 = __add( q0*f1, g1 );
		g2 = __adc( __umulhi(q0, f1), 0 );

		g1 = __add( q1*f0, g1 );
		g2 = __adc( __umulhi(q1, f0), g2 );

		g2 += q0*f2 + q1*f1 + q2*f0;


		/* STEP 4: i = (p - g) with code for correcting negatives */

		i0 = __sub( p0, g0 );
		i1 = __sbb( p1, g1 );
		i2 = __sbb( p2, g2 );

		if(i2 & 0x80000000) {
			i0 = __add( i0, f0);
			i1 = __adc( i1, f1);
			i2 = __adc( i2, f2);
		}
	} while( --count ); // End of countdown loop
}

#if (__CUDA_ARCH__ < 200)

__device__ static void process71
(
	u64 f, /* factor to be checked */
	u64 i, /* initial value */
	u64 m, /* multiplier - should be a power of i */
	u64 r, /* inverse of multiplier */
	u32 bmax, /* range of admissible candidate. only hits < bmax need to be considered. bmax should be odd. */
	u32 count, /* how many iterations thru the main loop. inversely proportional to the power of i used in multiplier */
	u32 *RES, /* global array to which successes, if any, should be returned */
	u32 si,
	u32 f_,
	u32 i_,
	u32 m_,
	u32 r_
)
{
	u32 i0 = (u32) i & MASK_24, i1 = (u32) (i >> 24) & MASK_24, i2 = (i_ << 16) | (u32) (i >> 48);
	u32 f0 = (u32) f & MASK_24, f1 = (u32) (f >> 24) & MASK_24, f2 = (f_ << 16) | (u32) (f >> 48);
	u32 m0 = (u32) m & MASK_24, m1 = (u32) (m >> 24) & MASK_24, m2 = (m_ << 16) | (u32) (m >> 48);
	u32 r0 = (r_ >> 24) | ((u32) (r << 8) & MASK_24), r1 = ((u32) (r >> 16) & MASK_24) , r2 = (u32) (r >> 40);

	do { /* loop down */

		/* only bothered about even valued candidates. choose either i or f-i */
		if (i0 & 1) {
			i0 = __sub(f0, i0) & MASK_24;
			i1 = __sbb(f1, i1) & MASK_24;
			i2 = __sbb(f2, i2);
		}

		/* check if we have an even-valued success */
		if( (i2 == 0) && ((i1 >> 8) == 0) && ((i0 | (i1<<24)) < bmax) ) {
			u32 count = atomicInc(RES, RESULT_BUFFER_COUNT) + 1;
			RES[count*2-1] = i0 | (i1<<24); /* TODO: Add array bounds check */
			RES[count*2] = si;
		}

		/* compute next iteration of i in 4 steps */

		/* step 1 - compute quotient = (HIGH72 of i*r) + 1 */
		u32 t1, t2, q0, q1, q2;

		t1 = UMUL24(i1, r0) + UMUL24(i0, r1) + UMUL24HI(i0, r0);
		t2 = UMUL24(i2,r0) + UMUL24(i1,r1) + UMUL24(i0,r2) + UMUL24HI(i1, r0) + UMUL24HI(i0, r1) + (t1 >> 24);
		q0 = UMUL24(i2,r1) + UMUL24(i1,r2) + UMUL24HI(i2,r0) + UMUL24HI(i1,r1) + UMUL24HI(i0,r2) + (t2 >> 24) + 1;
		q1 = UMUL24(i2,r2) + UMUL24HI(i2,r1) + UMUL24HI(i1,r2) + (q0 >> 24); q0 &= MASK_24;
		q2 = UMUL24HI(i2,r2) + (q1 >> 24); q1 &= MASK_24;


		/* step 2 - compute full product i * m */
		u32 p0, p1, p2;

		p0 = UMUL24(i0,m0);
		p1 = UMUL24(i1,m0) + UMUL24(i0,m1) + UMUL24HI(i0,m0);
		p2 = __umul24(i2,m0) + __umul24(i1,m1) + __umul24(i0,m2) + UMUL24HI(i1,m0) + UMUL24HI(i0,m1) + (p1 >> 24);

		p1 &= MASK_24;
		//p2 &= MASK_24;


		/* step 3 - compute qoutient*factor and subtract from P */
		u32 g0, g1, g2;

		g0 = UMUL24(q0,f0);
		g1 = UMUL24(q1,f0) + UMUL24(q0,f1) + UMUL24HI(q0,f0);
		g2 = __umul24(q2,f0) + __umul24(q1,f1) + __umul24(q0,f2) + UMUL24HI(q1,f0) + UMUL24HI(q0,f1) + (g1 >> 24);

		g1 &= MASK_24;
		//g2 &= MASK_24;


		/* STEP 4: i = (p - g) with branchless code for correcting negatives */

		i0 = __sub(p0, g0) & MASK_24;
		i1 = __sbb(p1, g1) & MASK_24;
		i2 = __sbb(p2, g2) & MASK_24;

		if(i2 & 0x800000) {
			i0 = __add(i0|~MASK_24, f0) & MASK_24;
			i1 = __adc(i1|~MASK_24, f1) & MASK_24;
			i2 = __adc(i2, f2) & MASK_24;
		}
	} while( --count ); // End of countdown loop
}

#endif

#else

__device__ static void process64
(
	u64 f, /* factor to be checked */
	u64 i, /* initial value */
	u64 m, /* multiplier - should be a power of i */
	u64 r, /* inverse of multiplier */
	u32 bmax, /* range of admissible candidate. only hits < bmax need to be considered. bmax should be odd. */
	u32 count, /* how many iterations thru the main loop. inversely proportional to the power of i used in multiplier */
	u32 *RES, /* global array to which successes, if any, should be returned */
	u32 si
)
{
	u32 i0 = (u32) i, i1 = (u32) (i >> 32);
	u32 f0 = (u32) f, f1 = (u32) (f >> 32);
	u32 m0 = (u32) m, m1 = (u32) (m >> 32);
	u32 r0 = (u32) r, r1 = (u32) (r >> 32);

	do { /* loop down */

		/* only bothered about even valued candidates. choose either i or f-i */
		if(i0 & 1) { /* tmp = FF..FF or 00..00 */
			i0 = __sub(f0, i0);
			i1 = __sbb(f1, i1);
		}

		/* check if we have an even-valued success */
		if( (i1 == 0) && (i0 < bmax) ) {
			u32 count = atomicInc(RES, RESULT_BUFFER_COUNT) + 1;
			RES[count*2-1] = i0; /* TODO: Add array bounds check */
			RES[count*2] = si;
		}

		/* compute next iteration of i in 4 steps */

		/* step 1 - compute quotient = (HIGH64 of i*r) + 1 */
		u32 t, q0, q1;

		t  = __umulhi(i0, r0);

		t  = __mad( i0, r1, t );
		q0 = __madchi( i0, r1, 1 );

		t  = __mad( i1, r0, t );
		q0 = __madchi( i1, r0, q0 );
		q1 = __adc( 0, 0 );

		q0 = __mad( i1, r1, q0 );
		q1 = __madchi( i1, r1, q1 );


		/* step 2 - compute full product i * m */
		u32 p0, p1, p2;

		p0 = i0*m0;
		p1 = __umulhi( i0, m0 );

		p1 = __mad( i0, m1, p1 );
		p2 = __madchi( i0, m1, 0 );

		p1 = __mad( i1, m0, p1 );
		p2 = __madchi( i1, m0, p2 );

		p2 = __mad( i1, m1, p2 );


		/* step 3 - compute qoutient*factor and subtract from P */
		u32 g0, g1, g2;

		g0 = q0*f0;
		g1 = __umulhi( q0, f0 );

		g1 = __mad( q0, f1, g1 );
		g2 = __madchi( q0, f1, 0 );

		g1 = __mad( q1, f0, g1 );
		g2 = __madchi( q1, f0, g2 );

		g2 = __mad( q1, f1, g2 );


		/* STEP 4: i = (p - g) with branchless code for correcting negatives */

		i0 = __sub( p0, g0 );
		i1 = __sbb( p1, g1 );
		p2 = __sbb( p2, g2 );

		if (p2) {
			i0 = __add( i0, f0 );
			i1 = __adc( i1, f1 );
		}
	} while( --count ); // End of countdown loop
}

__device__ static void process79
(
	u64 f, /* factor to be checked */
	u64 i, /* initial value */
	u64 m, /* multiplier - should be a power of i */
	u64 r, /* inverse of multiplier */
	u32 bmax, /* range of admissible candidate. only hits < bmax need to be considered. bmax should be odd. */
	u32 count, /* how many iterations thru the main loop. inversely proportional to the power of i used in multiplier */
	u32 *RES, /* global array to which successes, if any, should be returned */
	u32 si,
	u32 f_,
	u32 i_,
	u32 m_,
	u32 r_
)
{
	u32 i0 = (u32) i, i1 = (u32) (i >> 32), i2 = i_;
	u32 f0 = (u32) f, f1 = (u32) (f >> 32), f2 = f_;
	u32 m0 = (u32) m, m1 = (u32) (m >> 32), m2 = m_;
	u32 r0 = r_, r1 = (u32) r, r2 = (u32) (r >> 32);

	do { /* loop down */

		/* only bothered about even valued candidates. choose either i or f-i */
		if(i0 & 1) { /* tmp = FF..FF or 00..00 */
			i0 = __sub(f0, i0);
			i1 = __sbb(f1, i1);
			i2 = __sbb(f2, i2);
		}

		/* check if we have an even-valued success */
		if( ((i1|i2) == 0) && (i0 < bmax) ) {
			u32 count = atomicInc(RES, RESULT_BUFFER_COUNT) + 1;
			RES[count*2-1] = i0; /* TODO: Add array bounds check */
			RES[count*2] = si;
		}

		/* compute next iteration of i in 4 steps */

		/* step 1 - compute quotient = (HIGH96 of i*r) + 1 */
		u32 t1, t2, q0, q1, q2;

		//t0 = i0*r0;
		t1 = __umulhi(i0, r0);

		t1 = __mad( i0, r1, t1 );
		t2 = __madchi( i0, r1, 0 );

		t2 = __mad( i0, r2, t2 );
		q0 = __madchi( i0, r2, 1 );

		t1 = __mad( i1, r0, t1 );
		t2 = __madchi( i1, r0, t2 );
		q0 = __madc( i1, r2, q0 );
		q1 = __madchi( i1, r2, 0 );

		t2 = __mad( i1, r1, t2 );
		q0 = __madchi( i1, r1, q0 );
		q1 = __adc( q1, 0 );


		t2 = __mad( i2, r0, t2 );
		q0 = __madchi( i2, r0, q0 );
		q1 = __madc( i2, r2, q1 );
		q2 = __madchi( i2, r2, 0 );

		q0 = __mad( i2, r1, q0 );
		q1 = __madchi( i2, r1, q1 );
		q2 = __adc( q2, 0 );




		/* step 2 - compute full product i * m */
		u32 p0, p1, p2;

		p0 = i0*m0;
		p1 = __umulhi( i0, m0 );

		p1 = __mad( i0, m1, p1 );
		p2 = __madchi( i0, m1, 0 );

		p1 = __mad( i1, m0, p1 );
		p2 = __madchi( i1, m0, p2 );

		p2 += i0*m2 + i1*m1 + i2*m0;


		/* step 3 - compute qoutient*factor and subtract from P */
		u32 g0, g1, g2;

		g0 = q0*f0;
		g1 = __umulhi( q0, f0 );

		g1 = __mad( q0, f1, g1 );
		g2 = __madchi( q0, f1, 0 );

		g1 = __mad( q1, f0, g1 );
		g2 = __madchi( q1, f0, g2 );

		g2 += q0*f2 + q1*f1 + q2*f0;


		/* STEP 4: i = (p - g) with branchless code for correcting negatives */

		i0 = __sub( p0, g0 );
		i1 = __sbb( p1, g1 );
		i2 = __sbb( p2, g2 );

		if(i2 & 0x80000000) {
			i0 = __add( i0, f0);
			i1 = __adc( i1, f1);
			i2 = __adc( i2, f2);
		}
	} while( --count ); // End of countdown loop
}

#endif


__global__ void gfn_kernel63(u64 *fac_mult_ratio, u64 *init, u32 bmax, u32 count, u32 *RES, u32 init_fac_shift)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int shift_index = index >> init_fac_shift;

	process63(fac_mult_ratio[shift_index*3], init[index], fac_mult_ratio[shift_index*3+1], fac_mult_ratio[shift_index*3+2], bmax, count, RES, shift_index);
}

__global__ void gfn_kernel64(u64 *fac_mult_ratio, u64 *init, u32 bmax, u32 count, u32 *RES, u32 init_fac_shift)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int shift_index = index >> init_fac_shift;

	process64(fac_mult_ratio[shift_index*3], init[index], fac_mult_ratio[shift_index*3+1], fac_mult_ratio[shift_index*3+2], bmax, count, RES, shift_index);
}

__global__ void gfn_kernel71(u64 *fac_mult_ratio, u64 *init, u32 bmax, u32 count, u32 *RES, u32 init_fac_shift, u32 *fac_mult_ratio1, u32 *init1)
{
#if (__CUDA_ARCH__ < 200)
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int shift_index = index >> init_fac_shift;

	process71(
		fac_mult_ratio[shift_index*3],
		init[index],
		fac_mult_ratio[shift_index*3+1],
		fac_mult_ratio[shift_index*3+2],
		bmax,
		count,
		RES,
		shift_index,
		fac_mult_ratio1[shift_index*3],
		init1[index],
		fac_mult_ratio1[shift_index*3+1],
		fac_mult_ratio1[shift_index*3+2]
	);
#endif
}

__global__ void gfn_kernel79(u64 *fac_mult_ratio, u64 *init, u32 bmax, u32 count, u32 *RES, u32 init_fac_shift, u32 *fac_mult_ratio1, u32 *init1)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int shift_index = index >> init_fac_shift;

	process79(
		fac_mult_ratio[shift_index*3],
		init[index],
		fac_mult_ratio[shift_index*3+1],
		fac_mult_ratio[shift_index*3+2],
		bmax,
		count,
		RES,
		shift_index,
		fac_mult_ratio1[shift_index*3],
		init1[index],
		fac_mult_ratio1[shift_index*3+1],
		fac_mult_ratio1[shift_index*3+2]
	);
}
#endif // DEVICE_CUDA


static
void CUDA_error_exit(cudaError_t cudaError, int line)
{
	if(cudaError != cudaSuccess)
	{
		printf("GPU Error %d @ %d: %s \n", cudaError, line, cudaGetErrorString(cudaError));
		if(gd.d_Result != NULL) cudaFree(gd.d_Result);
		if(gd.h_Result != NULL) cudaFreeHost(gd.h_Result);
		if(gd.d_Init != NULL) cudaFree(gd.d_Init);
		if(gd.h_Init != NULL) cudaFreeHost(gd.h_Init);
		if(gd.d_Factor_Mult_Ratio!= NULL) cudaFree(gd.d_Factor_Mult_Ratio);
		if(gd.h_Factor_Mult_Ratio!= NULL) cudaFreeHost(gd.h_Factor_Mult_Ratio);
		if(gd.b_Factor_Mult_Ratio!= NULL) cudaFreeHost(gd.b_Factor_Mult_Ratio);
		if(gd.d_Init1 != NULL) cudaFree(gd.d_Init1);
		if(gd.h_Init1 != NULL) cudaFreeHost(gd.h_Init1);
		if(gd.d_Factor_Mult_Ratio1!= NULL) cudaFree(gd.d_Factor_Mult_Ratio1);
		if(gd.h_Factor_Mult_Ratio1!= NULL) cudaFreeHost(gd.h_Factor_Mult_Ratio1);
		if(gd.b_Factor_Mult_Ratio1!= NULL) cudaFreeHost(gd.b_Factor_Mult_Ratio1);
#ifdef DEVICE_OPENCL
		if (gd.device_ctx) ocl_cleanup_device(gd.device_ctx, true);
#endif

		exit(1);
	}
}

static
void usage_exit()
{
#if defined(DEVICE_OPENCL)
#define OPTIONS_EXTRA " [W]"
#else
#define OPTIONS_EXTRA ""
#endif

//	printf("GFN_Sieve <n> <bmax> <startp> <endp> [device number]\n");
	printf("\n\n" PROG_NAME " <n> <startp> <endp> [B<n>] [D<n>]" OPTIONS_EXTRA "\n\n");
	printf("n: n in b^2^n+1. n >= 15 and n <= 24\n");
//	printf("bmax: max range of b values. should be even < 2^31\n");
	printf("startp, endp: the p range. specify in PETA (10^15) values.\n");
	printf("\tvalid range is between 1 and 604462909\n");
	printf("B<n>: Block size. Optional. Controls the size of the block of\n");
	printf("\twork sent to GPU. Should be between 5 and 13 (inclusive).\n");
	printf("\tDefaults to 7 which is a good compromise. On slower systems,\n");
	printf("\tyou should reduce it to improve screen lag. On faster systems,\n");
	printf("\tyou can try increasing it to improve throughput\n");
	printf("\tEg:- Use B9 to increase the value to 9\n");
	printf("D<n>: Device Number. Optional. For systems with multiple GPU,	\n");
	printf("\tindicate the device to use. Defaults to 0 (i.e. Primary device).\n");
	printf("\tEg:- use D1 to specify running on second GPU\n");
#ifdef DEVICE_OPENCL
	printf("W: activate workaround for high CPU usage on NVidia.\n");
	printf("\tNote: in this mode you must run at least two copies of program\n");
	printf("\tto fully utilize GPU.\n");
#endif

	exit(1);
#undef OPTIONS_EXTRA
}

static
int isnumeric(char *arg)
{
	int c=0;
	if((arg[0] == '0') && arg[1])
		return 0; // leading zero not accepted

	while (*arg) {
		if(!isdigit(*arg))
			return 0; // not a digit character
		arg++;
		c++;
	}

	return (c > 0); // empty string is not acceptable
}


static
void parse_params(int argc, char * argv[])
{
	if(argc < 4)
	{
		printf("Incorrect number of arguments\n");
		usage_exit();
	}

	int n = atoi(argv[1]);
	if(errno || !isnumeric(argv[1]) || (n < 15) || (n > 24))
	{
		printf("Bad n value %s\n", argv[1]);
		usage_exit();
	}
	gd.n = n;

//	int bmax = atoi(argv[2]);
//	if(errno || !isnumeric(argv[2]) || (bmax < 2))
//	{
//		printf("Bad bmax value %s\n", argv[2]);
//		usage_exit();
//	}
//	if(bmax & 1)
//	{
//		bmax ^= 1;
//		printf("warning: even bmax expected. using %d\n", bmax);
//	}
//	gd.bmax = bmax;
	gd.bmax = 100000000; // 100M harcoded for PG

	int st = atoi(argv[2]);
	if(errno || !isnumeric(argv[2]) || (st < 1) || (st > 604462908))
	{
		printf("Bad startp value %s\n", argv[2]);
		usage_exit();
	}
	gd.startp_in_peta = st;

	int en = atoi(argv[3]);
	if(errno || !isnumeric(argv[3]) || (en <= st) || (en > 604462909))
	{
		printf("Bad endp value %s\n", argv[3]);
		usage_exit();
	}
	gd.endp_in_peta = en;

	gd.device_number = 0;
	gd.b_blocks_per_grid = 7; // A sensible default

	// Positional arguments are done. Now named arguments

	for (int i=4; i < argc; i++) {
		switch(argv[i][0]) {
			case 'b':
			case 'B': // Block count

			{
				int bn = atoi(argv[i]+1);

				if(errno || !isnumeric(argv[i]+1) || bn < 5 || bn > 13)
				{
					printf("Bad value for block size '%s'\n", argv[i]+1);
					usage_exit();
				}
				gd.b_blocks_per_grid = bn;
				break;
			}

			case 'd':
			case 'D': // Device number

			{
				int dn = atoi(argv[i]+1);

				if(errno || !isnumeric(argv[i]+1) || dn < 0)
				{
					printf("Bad value for device number '%s'\n", argv[i]+1);
					usage_exit();
				}
				gd.device_number = dn;
				break;
			}

#ifdef DEVICE_OPENCL
			case 'w':
			case 'W': // NVidia workaround

			{
				gd.use_nvidia_workaround = 1;
				break;
			}
#endif

			default:
				printf("Unknown argument '%s'\n", argv[i]);
				usage_exit();

		}

	}
	gd.N = 1 << gd.n;
	gd.N1 = gd.N << 1;
	gd.inv_N1 = 1.0/(double)gd.N1;
	gd.stat = 0;
	gd.factorcount = 0;

	printf("GFN Sieve for k^%u+1 [k == 2 to %u]\n\n", gd.N, gd.bmax);

	sprintf(gd.fact, "f%u_%uP_%uP.txt", gd.n, gd.startp_in_peta, gd.endp_in_peta);
	printf("Using factor file '%s'\n", gd.fact);

	sprintf(gd.ckpt, "c%u_%uP_%uP.txt", gd.n, gd.startp_in_peta, gd.endp_in_peta);
	printf("Using checkpoint file '%s'\n", gd.ckpt);
}


int main(int argc, char *argv[])
{
	printf(PROG_TITLE "\n\n");
#ifdef DEVICE_SIMULATION
	printf("SIMULATION/BENCHMARK MODE - WILL NOT FIND ANY FACTORS\n\n");
#endif

	gd.d_Result = NULL;
	gd.h_Result = NULL;
	gd.d_Init  = NULL;
	gd.h_Init  = NULL;
	gd.d_Factor_Mult_Ratio= NULL;
	gd.h_Factor_Mult_Ratio= NULL;
	gd.b_Factor_Mult_Ratio= NULL;

	parse_params(argc, argv);

	U64 startk, endk, resumek, crossover63, crossover64, crossover71;


	startk = peta_to_k(gd.startp_in_peta); // approximation - might do 1 more k than necessary
	endk = peta_to_k(gd.endp_in_peta);
	crossover63 = 1; crossover63 <<= 63 - (gd.n+1);
	crossover64 = 1; crossover64 <<= 64 - (gd.n+1);
	crossover71 = 1; crossover71 <<= 71 - (gd.n+1);

	resumek = read_checkpoint(startk, endk);
	if (resumek) {
		if (resumek > endk) {
			printf("This range is already done\n");
			return 0;
		}
		else {
			HALF t; initHALF(t, resumek); shlHALF(t, gd.n+1); t[0]++;
			printf("Resuming from checkpoint value %s\n", HALF2Str(t));
			startk = resumek;
		}
	}
	else {
		write_factor_header();
	}

	init_sieve_primes(startk);

	term_requested = 0;

#ifdef PLATFORM_WIN32
	SetConsoleCtrlHandler( CtrlHandler, TRUE );
#endif
#ifdef PLATFORM_LINUX
	signal( SIGINT, my_sig_handler );
	signal( SIGTERM, my_sig_handler );
#endif

#define B_THREADS_PER_BLOCK        8
#define THREADS_PER_BLOCK        (1 << B_THREADS_PER_BLOCK)
#define B_MAX_ITER_PER_CAND        15

	// Compute dimensions of the CUDA kernel
	u32 b_iter_per_fac = (gd.n - 1);
	//u32   iter_per_fac = gd.N >> 1;

	u32 b_iter_per_cand = B_MAX_ITER_PER_CAND;   // New larger default value of b_iter_per_cand
	gd.b_blocks_per_grid -= 3; // Older b_iter_per_cand was 12. To keep grid work the same, reduce block count.

	// For lower n's there is not enough iterations available.
	u32 b_max_iter_per_cand = b_iter_per_fac; // For Cyclo sieve, b_max_iter_per_cand = b_iter_per_fac-1
	if (b_max_iter_per_cand < B_MAX_ITER_PER_CAND)
	{
		// Reduce the # iterations to the max available
		b_iter_per_cand = b_max_iter_per_cand;
		// Increase the blocks to compensate.
		gd.b_blocks_per_grid += (B_MAX_ITER_PER_CAND - b_iter_per_cand);
	}
	gd.blocks_per_grid = 1 << gd.b_blocks_per_grid;

	u32 b_cand_per_grid = B_THREADS_PER_BLOCK + gd.b_blocks_per_grid;
	u32 b_cand_per_fac = b_iter_per_fac - b_iter_per_cand;

	u32 cand_per_grid = 1 << b_cand_per_grid;
	u32 iter_per_cand = 1 << b_iter_per_cand;
	u32 cand_per_fac = 1 << b_cand_per_fac;
	u32 fac_per_grid = cand_per_grid / cand_per_fac;

#ifdef DEVICE_CUDA
	// Hoping to avoid CPU busy-wait behavior. Might also improve screen responsiveness
	// Remember: SetDeviceFlags should be done before SetDevice!!
	CUDA_error_exit( cudaSetDeviceFlags( BLOCKING_SYNC ), __LINE__ );

	// Select device. Verify that we can use it.
	CUDA_error_exit( cudaSetDevice(gd.device_number), __LINE__ );
	CUDA_error_exit( cudaGetDeviceProperties(&gd.device_info, gd.device_number), __LINE__ );

	if((gd.device_info.major == 1) && (gd.device_info.minor == 0)) {
		printf("Need at least CC 1.1 card\n");
		return 1;
	}
	if(gd.device_info.major == 1) {
		printf("\n\nWARNING! Compute 1.x card detected.\n"
			"For optimal performance, Compute 2.x or better is recommended\n\n");
	}
#endif
#ifdef DEVICE_OPENCL
	int nDev = ocl_initialize_devices();
	if (nDev <= 0)
	{
		printf("No compatible OpenCL devices found\n");
		return 1;
	}
	if (gd.device_number >= nDev)
	{
		printf("Cannot use device index %d, only %d devices detected\n", gd.device_number, nDev);
		return 1;
	}
	gd.device_ctx = ocl_get_context(gd.device_number);
	if (gd.device_ctx == NULL || !gd.device_ctx->active)
	{
		printf("Failed to activate device %d!\n", gd.device_number);
		return 1;
	}
	CUDA_error_exit( ocl_preconfigure(gd.device_ctx), __LINE__ );
	cl_event event_in_progress = NULL;
#endif

	CUDA_error_exit( cudaHostAlloc((void**)&gd.b_Factor_Mult_Ratio, fac_per_grid * sizeof(u64) * 3, 0), __LINE__ );
	CUDA_error_exit( cudaHostAlloc((void**)&gd.h_Factor_Mult_Ratio, fac_per_grid * sizeof(u64) * 3, 0), __LINE__ );
	CUDA_error_exit( cudaMalloc_ro((void**)&gd.d_Factor_Mult_Ratio, fac_per_grid * sizeof(u64) * 3), __LINE__ );

	CUDA_error_exit( cudaHostAlloc((void**)&gd.h_Init, cand_per_grid * sizeof(u64), 0), __LINE__ );
	CUDA_error_exit( cudaMalloc_ro((void**)&gd.d_Init, cand_per_grid * sizeof(u64)), __LINE__ );

	CUDA_error_exit( cudaHostAlloc((void**)&gd.h_Result, RESULT_BUFFER_SIZE * sizeof(u32), 0), __LINE__ );
	CUDA_error_exit( cudaMalloc_rw((void**)&gd.d_Result, RESULT_BUFFER_SIZE * sizeof(u32)), __LINE__ );

	CUDA_error_exit( cudaHostAlloc((void**)&gd.b_Factor_Mult_Ratio1, fac_per_grid * sizeof(u32) * 3, 0), __LINE__ );
	CUDA_error_exit( cudaHostAlloc((void**)&gd.h_Factor_Mult_Ratio1, fac_per_grid * sizeof(u32) * 3, 0), __LINE__ );
	CUDA_error_exit( cudaMalloc_ro((void**)&gd.d_Factor_Mult_Ratio1, fac_per_grid * sizeof(u32) * 3), __LINE__ );

	CUDA_error_exit( cudaHostAlloc((void**)&gd.h_Init1, cand_per_grid * sizeof(u32), 0), __LINE__ );
	CUDA_error_exit( cudaMalloc_ro((void**)&gd.d_Init1, cand_per_grid * sizeof(u32)), __LINE__ );

	gd.starttime = MY_TIME();
	gd.factorsInBuffer = 0;
	gd.candsInBuffer = 0;
	gd.h_Result[0] = 0; // Only clear the count

	bool kernel_in_progress = false;
	U64 last_processed_k = 0;

	int core = 0;

	for(;;) {
		sieve_iteration();

		for(U32 j=0; j < SIEVE_SIZE*32; j++) {
			if(bit_test(sieve, j)) {
				gd.k = startk + j;
				// We're not checking end of range at this step.
				// This might process a few k's that are past the range. That is fine.
				// There are no safety issues because we're limiting the valid range to slightly under what we can handle (2^79)

				processk(cand_per_fac);

				// Do we have a full batch of candidates yet to process?
				if (gd.candsInBuffer == cand_per_grid) {

					// We're trying to do candiate preparation (CPU) and factor finding (GPU) in a
					// piplelined fashion, to maximize thruput.
					if( kernel_in_progress ) {
#ifdef DEVICE_CUDA
						// Sync for the previous kernel. Not really needed, as the following Memcpy will do it anyways.
						CUDA_error_exit( SYNC_CALL(), __LINE__ );

						CUDA_error_exit( cudaMemcpy(gd.h_Result, gd.d_Result, RESULT_BUFFER_SIZE * sizeof(u32), cudaMemcpyDeviceToHost), __LINE__ );

						CUDA_error_exit( SYNC_CALL(), __LINE__ );
#endif // DEVICE_CUDA
#ifdef DEVICE_OPENCL
						if (gd.use_nvidia_workaround)
						{
#if 1
							for (;;)
							{
								cl_int ev_status;

								CUDA_error_exit( ocl_diagnose( clGetEventInfo(event_in_progress, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ev_status), &ev_status, NULL), "clGetEventInfo", gd.device_ctx ), __LINE__ );
								if (ev_status < CL_COMPLETE)
									CUDA_error_exit(ev_status, __LINE__);
								if (ev_status == CL_COMPLETE)
									break;
								Sleep(1);
							}
#else // does not seems to work. either 100% CPU usage either very low GPU load and deadlock on termination.
							cl_int ev_status;

							ResetEvent(gd.ocl_system_event);
							CUDA_error_exit( ocl_diagnose( clSetEventCallback(event_in_progress, CL_COMPLETE, ocl_event_callback, NULL), "clSetEventCallback", gd.device_ctx ), __LINE__ );
							CUDA_error_exit( ocl_diagnose( clGetEventInfo(event_in_progress, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ev_status), &ev_status, NULL), "clGetEventInfo", gd.device_ctx ), __LINE__ );
							if (ev_status < CL_COMPLETE)
								CUDA_error_exit(ev_status, __LINE__);
							if (ev_status != CL_COMPLETE)
								WaitForSingleObject(gd.ocl_system_event, INFINITE);
#endif
						}
						/* otherwise, clEnqueueReadBuffer will automatically wait for completion of kernel since queue has "in-order" type */
						CUDA_error_exit( ocl_diagnose( clEnqueueReadBuffer(gd.device_ctx->cmdQueue, gd.d_Result, CL_TRUE, 0, RESULT_BUFFER_SIZE * sizeof(u32), gd.h_Result, 0, NULL, NULL), "clEnqueueReadBuffer", gd.device_ctx ), __LINE__ );
						CUDA_error_exit( ocl_diagnose( clReleaseEvent(event_in_progress), "clReleaseEvent", gd.device_ctx ), __LINE__ );
#endif
						U32 factor_count = gd.h_Result[0];
						if(factor_count > 0) {
#ifdef DEVICE_OPENCL
							/* OpenCL does not have buffer size check in atomic_inc.
							   Check it here and hope that we dind't crashed whole PC.
							*/
							if (factor_count > RESULT_BUFFER_COUNT)
							{
								printf( "\nFATAL ERROR: result buffer overflow.\n\n"
									"If error is repeatable at same 'p', try to decrease 'B' parameter\n"
									"and contact author. Otherwise, it may be GPU overclock/overheat problem.\n"
									);
								CUDA_error_exit(-9999, __LINE__);
							}
#endif
							// printf("\nBatch count: %u\n", factor_count);
							write_factor_batch( factor_count );
						}

						gd.stat += fac_per_grid;
						gd.factorcount += factor_count;
						HALF tmpf;
						initHALF2(tmpf, gd.b_Factor_Mult_Ratio[(fac_per_grid-1)*3], gd.b_Factor_Mult_Ratio1[(fac_per_grid-1)*3]);

						print_status( tmpf );
						write_checkpoint(last_processed_k+1); // might or might not write an actual checkpoint.

						// The last kernel batch has been successfully handled.
						// Before proceeding to the next batch... Should we?
						// It is slightly inefficient in that, the CPU has been wasted to create a batch
						// only for it to be abandoned. But we're looking at a few milliseconds of CPU time,
						// and gains some simplification in code organization
						if(term_requested || (last_processed_k >= endk)) {
							write_checkpoint(last_processed_k+1, true); // Forced write of checkpoint
							goto freeresources;
						}
					}

					// Execute a new batch
					// The memcpy's are synchronous. The kernel is asynchronous.
					gd.h_Result[0] = 0; // Only clear the counter
					CUDA_error_exit( cudaMemcpy_htd(gd.d_Result, gd.h_Result, 1 * sizeof(u32), cudaMemcpyHostToDevice), __LINE__ );
					CUDA_error_exit( cudaMemcpy_htd(gd.d_Init  , gd.h_Init  , cand_per_grid * sizeof(u64), cudaMemcpyHostToDevice), __LINE__ );
					CUDA_error_exit( cudaMemcpy_htd(gd.d_Factor_Mult_Ratio, gd.h_Factor_Mult_Ratio,  fac_per_grid * sizeof(u64) * 3, cudaMemcpyHostToDevice), __LINE__ );
					CUDA_error_exit( cudaMemcpy_hth(gd.b_Factor_Mult_Ratio, gd.h_Factor_Mult_Ratio,  fac_per_grid * sizeof(u64) * 3, cudaMemcpyHostToHost  ), __LINE__ );
					CUDA_error_exit( cudaMemcpy_hth(gd.b_Factor_Mult_Ratio1, gd.h_Factor_Mult_Ratio1,  fac_per_grid * sizeof(u32) * 3, cudaMemcpyHostToHost  ), __LINE__ );

#ifdef DEVICE_CUDA
					if(gd.device_info.major == 1) {
						if (gd.k < crossover63) {
							core = 63;
							gfn_kernel63<<<gd.blocks_per_grid, THREADS_PER_BLOCK>>>
							(gd.d_Factor_Mult_Ratio
							, gd.d_Init
							, gd.bmax+1
							, iter_per_cand
							, gd.d_Result
							, b_cand_per_fac);
						}
						else if (gd.k < crossover71) {
							if(core == 63) {
								printf("Switching to 71 bit core...                                                   \n\n");
							}
								core = 71;
					CUDA_error_exit( cudaMemcpy(gd.d_Init1  , gd.h_Init1  , cand_per_grid * sizeof(u32), cudaMemcpyHostToDevice), __LINE__ );
					CUDA_error_exit( cudaMemcpy(gd.d_Factor_Mult_Ratio1, gd.h_Factor_Mult_Ratio1,  fac_per_grid * sizeof(u32) * 3, cudaMemcpyHostToDevice), __LINE__ );
							gfn_kernel71<<<gd.blocks_per_grid, THREADS_PER_BLOCK>>>
							(gd.d_Factor_Mult_Ratio
							, gd.d_Init
							, gd.bmax+1
							, iter_per_cand
							, gd.d_Result
							, b_cand_per_fac
							, gd.d_Factor_Mult_Ratio1
							, gd.d_Init1);
						}
						else {
							if(core == 71) {
								printf("Switching to 79 bit core...                                                   \n\n");
							}
							core = 79;
					CUDA_error_exit( cudaMemcpy(gd.d_Init1  , gd.h_Init1  , cand_per_grid * sizeof(u32), cudaMemcpyHostToDevice), __LINE__ );
					CUDA_error_exit( cudaMemcpy(gd.d_Factor_Mult_Ratio1, gd.h_Factor_Mult_Ratio1,  fac_per_grid * sizeof(u32) * 3, cudaMemcpyHostToDevice), __LINE__ );
							gfn_kernel79<<<gd.blocks_per_grid, THREADS_PER_BLOCK>>>
							(gd.d_Factor_Mult_Ratio
							, gd.d_Init
							, gd.bmax+1
							, iter_per_cand
							, gd.d_Result
							, b_cand_per_fac
							, gd.d_Factor_Mult_Ratio1
							, gd.d_Init1);
						}
					}
					else {
						if (gd.k < crossover63) {
							core = 63;
							gfn_kernel63<<<gd.blocks_per_grid, THREADS_PER_BLOCK>>>
							(gd.d_Factor_Mult_Ratio
							, gd.d_Init
							, gd.bmax+1
							, iter_per_cand
							, gd.d_Result
							, b_cand_per_fac);
						}
						else if (gd.k < crossover64) {
							if(core == 63) {
								printf("Switching to 64 bit core...                                                   \n\n");
							}
							core = 64;
							gfn_kernel64<<<gd.blocks_per_grid, THREADS_PER_BLOCK>>>
							(gd.d_Factor_Mult_Ratio
							, gd.d_Init
							, gd.bmax+1
							, iter_per_cand
							, gd.d_Result
							, b_cand_per_fac);
						}
						else {
							if(core == 64) {
								printf("Switching to 79 bit core...                                                   \n\n");
							}
							core = 79;
					CUDA_error_exit( cudaMemcpy(gd.d_Init1  , gd.h_Init1  , cand_per_grid * sizeof(u32), cudaMemcpyHostToDevice), __LINE__ );
					CUDA_error_exit( cudaMemcpy(gd.d_Factor_Mult_Ratio1, gd.h_Factor_Mult_Ratio1,  fac_per_grid * sizeof(u32) * 3, cudaMemcpyHostToDevice), __LINE__ );
							gfn_kernel79<<<gd.blocks_per_grid, THREADS_PER_BLOCK>>>
							(gd.d_Factor_Mult_Ratio
							, gd.d_Init
							, gd.bmax+1
							, iter_per_cand
							, gd.d_Result
							, b_cand_per_fac
							, gd.d_Factor_Mult_Ratio1
							, gd.d_Init1);
						}
					}
#endif // DEVICE_CUDA
#ifdef DEVICE_OPENCL
					if (gd.k < crossover63)
					{
						if (core != 63)
						{
							printf("\nUsing 63 bit core.\n\n");
							core = 63;
						}
						CUDA_error_exit( ocl_execute_core(core, gd.device_ctx, &event_in_progress
							, gd.blocks_per_grid * THREADS_PER_BLOCK
							, gd.d_Factor_Mult_Ratio
							, gd.d_Init
							, gd.bmax+1
							, iter_per_cand
							, gd.d_Result
							, b_cand_per_fac
							, NULL
							, NULL
							), __LINE__
						);
					}
					else if (gd.k < crossover64)
					{
						if (core != 64)
						{
							printf("\nUsing 64 bit core.\n\n");
							core = 64;
						}
						CUDA_error_exit( ocl_execute_core(core, gd.device_ctx, &event_in_progress
							, gd.blocks_per_grid * THREADS_PER_BLOCK
							, gd.d_Factor_Mult_Ratio
							, gd.d_Init
							, gd.bmax+1
							, iter_per_cand
							, gd.d_Result
							, b_cand_per_fac
							, NULL
							, NULL
							), __LINE__
						);
					}
					else
					{
						if (core != 79)
						{
							printf("\nUsing 79 bit core.\n\n");
							core = 79;
						}
						CUDA_error_exit( cudaMemcpy_htd(gd.d_Init1  , gd.h_Init1  , cand_per_grid * sizeof(u32), cudaMemcpyHostToDevice), __LINE__ );
						CUDA_error_exit( cudaMemcpy_htd(gd.d_Factor_Mult_Ratio1, gd.h_Factor_Mult_Ratio1,  fac_per_grid * sizeof(u32) * 3, cudaMemcpyHostToDevice), __LINE__ );
						CUDA_error_exit( ocl_execute_core(core, gd.device_ctx, &event_in_progress
							, gd.blocks_per_grid * THREADS_PER_BLOCK
							, gd.d_Factor_Mult_Ratio
							, gd.d_Init
							, gd.bmax+1
							, iter_per_cand
							, gd.d_Result
							, b_cand_per_fac
							, gd.d_Factor_Mult_Ratio1
							, gd.d_Init1
							), __LINE__
						);
					}
#endif // DEVICE_OPENCL
					kernel_in_progress = true;
					last_processed_k = gd.k; // this is the last k in the currently executing batch
					gd.candsInBuffer = 0;
					gd.factorsInBuffer = 0;

				}
			}
		}
		startk += SIEVE_BITS;
	}




freeresources:
	cudaFree(gd.d_Init1);
	cudaFreeHost(gd.h_Init1);
	cudaFree(gd.d_Factor_Mult_Ratio1);
	cudaFreeHost(gd.h_Factor_Mult_Ratio1);
	cudaFreeHost(gd.b_Factor_Mult_Ratio1);
	cudaFree(gd.d_Result);
	cudaFreeHost(gd.h_Result);
	cudaFree(gd.d_Init);
	cudaFreeHost(gd.h_Init);
	cudaFree(gd.d_Factor_Mult_Ratio);
	cudaFreeHost(gd.h_Factor_Mult_Ratio);
	cudaFreeHost(gd.b_Factor_Mult_Ratio);
#ifdef DEVICE_OPENCL
	ocl_cleanup_device(gd.device_ctx, true);
#endif

	return 0;
}
