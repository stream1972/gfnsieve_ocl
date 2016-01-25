#define u32 uint
#define i32 int
#define u64 ulong
#define __umulhi mul_hi

#ifdef __NV_CL_C_VERSION

/* Functions which set carry flag */
u32 __add_set(u32 a, u32 b)
{
	u32 r;
	asm volatile ("add.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
	return r;
}

u32 __adc_set(u32 a, u32 b)
{
	u32 r;
	asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
	return r;
}

u32 __sub_set(u32 a, u32 b)
{
	u32 r;
	asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
	return r;
}

u32 __sbb_set(u32 a, u32 b)
{
	u32 r;
	asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
	return r;
}

u32 __mad_set(u32 a, u32 b, u32 c)
{
	u32 r;
	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
	return r;
}

u32 __madc_set(u32 a, u32 b, u32 c)
{
	u32 r;
	asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
	return r;
}

u32 __madchi_set(u32 a, u32 b, u32 c)
{
	u32 r;
	asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r" (r) : "r" (a) , "r" (b), "r" (c));
	return r;
}

/* Functions which NOT set final carry flags. Probably no performance loss in setting it,
   so just reuse functions above
 */
#define __adc_none    __adc_set
#define __sbb_none    __sbb_set
#define __mad_none    __mad_set
#define __madchi_none __madchi_set

#define DEFINE_CARRY

#else // __NV_CL_C_VERSION

#warning No built-in carry flag support known, using plain C code
#define DEFINE_CARRY           u64 carry
#define __add_set(a, b)        (u32)(carry = (u64)(a) + (b))
#define __adc_set(a, b)        (u32)(carry = (u64)(a) + (b) + (carry >> 32))
#define __adc_none(a, b)       (a) + (b) + (u32)(carry >> 32)
#define __sub_set(a, b)        (u32)(carry = (u64)(a) - (b))
#define __sbb_set(a, b)        (u32)(carry = (u64)(a) - (b) - (u32)(-((i32)(carry >> 32)))) // use only after __sub (carry must be 0xFFFFFFF)
#define __sbb_none(a, b)       (a) - (b) + (i32)(carry >> 32)                      // use only after __sub (carry must be 0xFFFFFFF)
#define __mad_set(a, b, c)     (u32)(carry = (u64)((a) * (b)) + (c))
#define __madc_set(a, b, c)    (u32)(carry = (u64)((a) * (b)) + (c) + (carry >> 32))
#define __mad_none(a, b, c)    (a) * (b) + (c)
#define __madchi_set(a, b, c)  (u32)(carry = (u64)mul_hi((a), (b)) + (c) + (carry >> 32))
#define __madchi_none(a, b, c) mul_hi((a), (b)) + (c) + (carry >> 32)

#endif // __NV_CL_C_VERSION

__kernel void process79(__global u64 *fac_mult_ratio,  __global u64 *init, __global u32 *RES,
			__global u32 *fac_mult_ratio1, __global u32 *init1)
{
	u32 index         = get_global_id(0);
	u32 si            = index >> init_fac_shift;
	u32 shift_index_3 = si * 3;

	u64 f; /* factor to be checked */
	u64 i; /* initial value */
	u64 m; /* multiplier - should be a power of i */
	u64 r; /* inverse of multiplier */
	/* (param) u32 bmax; */ /* range of admissible candidate. only hits < bmax need to be considered. bmax should be odd. */
	u32 count; /* how many iterations thru the main loop. inversely proportional to the power of i used in multiplier */
	/* (param) u32 *RES; */ /* global array to which successes, if any, should be returned */
	u32 f_;
	u32 i_;
	u32 m_;
	u32 r_;

	f = fac_mult_ratio[shift_index_3];
	i = init[index];
	m = fac_mult_ratio[shift_index_3+1];
	r = fac_mult_ratio[shift_index_3+2];
	f_ = fac_mult_ratio1[shift_index_3];
	i_ = init1[index];
	m_ = fac_mult_ratio1[shift_index_3+1];
	r_ = fac_mult_ratio1[shift_index_3+2];
	count = param_count;

	DEFINE_CARRY;

	u32 i0 = (u32) i, i1 = (u32) (i >> 32), i2 = i_;
	u32 f0 = (u32) f, f1 = (u32) (f >> 32), f2 = f_;
	u32 m0 = (u32) m, m1 = (u32) (m >> 32), m2 = m_;
	u32 r0 = r_, r1 = (u32) r, r2 = (u32) (r >> 32);

	do { /* loop down */

		/* only bothered about even valued candidates. choose either i or f-i */
		if(i0 & 1) { /* tmp = FF..FF or 00..00 */
			i0 = __sub_set(f0, i0);
			i1 = __sbb_set(f1, i1);
			i2 = __sbb_none(f2, i2);
		}

		/* check if we have an even-valued success */
		if( ((i1|i2) == 0) && (i0 < bmax) ) {
			u32 count = atomic_inc(RES) + 1;
			RES[count*2-1] = i0; /* TODO: Add array bounds check */
			RES[count*2] = si;
		}

		/* compute next iteration of i in 4 steps */

		/* step 1 - compute quotient = (HIGH96 of i*r) + 1 */
		u32 t1, t2, q0, q1, q2;

		//t0 = i0*r0;
		t1 = __umulhi(i0, r0);

		t1 = __mad_set( i0, r1, t1 );
		t2 = __madchi_none( i0, r1, 0 );

		t2 = __mad_set( i0, r2, t2 );
		q0 = __madchi_none( i0, r2, 1 );

		t1 = __mad_set( i1, r0, t1 );
		t2 = __madchi_set( i1, r0, t2 );
		q0 = __madc_set( i1, r2, q0 );
		q1 = __madchi_none( i1, r2, 0 );

		t2 = __mad_set( i1, r1, t2 );
		q0 = __madchi_set( i1, r1, q0 );
		q1 = __adc_none( q1, 0 );


		t2 = __mad_set( i2, r0, t2 );
		q0 = __madchi_set( i2, r0, q0 );
		q1 = __madc_set( i2, r2, q1 );
		q2 = __madchi_none( i2, r2, 0 );

		q0 = __mad_set( i2, r1, q0 );
		q1 = __madchi_set( i2, r1, q1 );
		q2 = __adc_none( q2, 0 );


		/* step 2 - compute full product i * m */
		u32 p0, p1, p2;

		p0 = i0*m0;
		p1 = __umulhi( i0, m0 );

		p1 = __mad_set( i0, m1, p1 );
		p2 = __madchi_none( i0, m1, 0 );

		p1 = __mad_set( i1, m0, p1 );
		p2 = __madchi_none( i1, m0, p2 );

		p2 += i0*m2 + i1*m1 + i2*m0;


		/* step 3 - compute qoutient*factor and subtract from P */
		u32 g0, g1, g2;

		g0 = q0*f0;
		g1 = __umulhi( q0, f0 );

		g1 = __mad_set( q0, f1, g1 );
		g2 = __madchi_none( q0, f1, 0 );

		g1 = __mad_set( q1, f0, g1 );
		g2 = __madchi_none( q1, f0, g2 );

		g2 += q0*f2 + q1*f1 + q2*f0;


		/* STEP 4: i = (p - g) with branchless code for correcting negatives */

		i0 = __sub_set( p0, g0 );
		i1 = __sbb_set( p1, g1 );
		i2 = __sbb_none( p2, g2 );

		if(i2 & 0x80000000) {
			i0 = __add_set( i0, f0 );
			i1 = __adc_set( i1, f1 );
			i2 = __adc_none( i2, f2 );
		}
	} while( --count ); // End of countdown loop
}
