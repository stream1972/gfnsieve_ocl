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
#define __adc_none(a, b)       (a) + (b) + (u32)(carry >> 32)
#define __sub_set(a, b)        (u32)(carry = (u64)(a) - (b))
#define __sbb_set(a, b)        (u32)(carry = (u64)(a) - (b) - (u32)(-((i32)(carry >> 32)))) // use only after __sub (carry must be 0xFFFFFFF)
#define __sbb_none(a, b)       (a) - (b) + (i32)(carry >> 32)                      // use only after __sub (carry must be 0xFFFFFFF)
#define __mad_set(a, b, c)     (u32)(carry = (u64)((a) * (b)) + (c))
#define __mad_none(a, b, c)    (a) * (b) + (c)
#define __madchi_set(a, b, c)  (u32)(carry = (u64)mul_hi((a), (b)) + (c) + (carry >> 32))
#define __madchi_none(a, b, c) mul_hi((a), (b)) + (c) + (carry >> 32)

#endif // __NV_CL_C_VERSION

__kernel void process64(__global u64 *fac_mult_ratio, __global u64 *init, __global u32 *RES)
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

	f = fac_mult_ratio[shift_index_3];
	i = init[index];
	m = fac_mult_ratio[shift_index_3+1];
	r = fac_mult_ratio[shift_index_3+2];
	count = param_count;

#if 1
	/* Use assembly math-with-carry (NVIDIA)
	 * or emulate carry flag in high word of 64-bit operation
	 */
	DEFINE_CARRY;

	u32 i0 = (u32) i, i1 = (u32) (i >> 32);
	u32 f0 = (u32) f, f1 = (u32) (f >> 32);
	u32 m0 = (u32) m, m1 = (u32) (m >> 32);
	u32 r0 = (u32) r, r1 = (u32) (r >> 32);

	do { /* loop down */

		/* only bothered about even valued candidates. choose either i or f-i */
		if(i0 & 1) { /* tmp = FF..FF or 00..00 */
			i0 = __sub_set(f0, i0);
			i1 = __sbb_none(f1, i1);
		}

		/* check if we have an even-valued success */
		if( (i1 == 0) && (i0 < bmax) ) {
			u32 count = atomic_inc(RES) + 1;
			RES[count*2-1] = i0; /* TODO: Add array bounds check */
			RES[count*2] = si;
		}

		/* compute next iteration of i in 4 steps */

		/* step 1 - compute quotient = (HIGH64 of i*r) + 1 */
		u32 t, q0, q1;

		t  = __umulhi(i0, r0);

		t  = __mad_set( i0, r1, t );
		q0 = __madchi_none( i0, r1, 1 );

		t  = __mad_set( i1, r0, t );
		q0 = __madchi_set( i1, r0, q0 );
		q1 = __adc_none( 0, 0 );

		q0 = __mad_set( i1, r1, q0 );
		q1 = __madchi_none( i1, r1, q1 );


		/* step 2 - compute full product i * m */
		u32 p0, p1, p2;

		p0 = i0*m0;
		p1 = __umulhi( i0, m0 );

		p1 = __mad_set( i0, m1, p1 );
		p2 = __madchi_none( i0, m1, 0 );

		p1 = __mad_set( i1, m0, p1 );
		p2 = __madchi_none( i1, m0, p2 );

		p2 = __mad_none( i1, m1, p2 );


		/* step 3 - compute qoutient*factor and subtract from P */
		u32 g0, g1, g2;

		g0 = q0*f0;
		g1 = __umulhi( q0, f0 );

		g1 = __mad_set( q0, f1, g1 );
		g2 = __madchi_none( q0, f1, 0 );

		g1 = __mad_set( q1, f0, g1 );
		g2 = __madchi_none( q1, f0, g2 );

		g2 = __mad_none( q1, f1, g2 );


		/* STEP 4: i = (p - g) with branchless code for correcting negatives */

		i0 = __sub_set( p0, g0 );
		i1 = __sbb_set( p1, g1 );
		p2 = __sbb_none( p2, g2 );

		if (p2) {
			i0 = __add_set( i0, f0 );
			i1 = __adc_none( i1, f1 );
		}
	} while( --count ); // End of countdown loop

#else
	/* No carry flag usage/emulation, do simple 64-bit math operations when possible
	 * (optimization is up to compiler), but need to call mul_hi(u64, u64) to get
	 * high part of 128-bit multiplication result.
	 * According to testing on GTX 750ti and some old R2xx, this code is SLOWER then
	 * even emulation of carry flag. So it's disabled for now.
	 */

	do { /* loop down */

		/* only bothered about even valued candidates. choose either i or f-i */
		if (i & 1) { /* tmp = FF..FF or 00..00 */
			i = f-i;
		}

		/* check if we have an even-valued success */
		if (i < bmax) {
			u32 count = atomic_inc(RES) + 1;
			RES[count*2-1] = (u32)i; /* TODO: Add array bounds check */
			RES[count*2] = si;
		}

		/* compute next iteration of i in 4 steps */

		/* step 1 - compute quotient = (HIGH64 of i*r) + 1 */
		u64 q = __umulhi(i, r) + 1;

		/* step 2 - compute full product i * m */
		u64 p  = i * m;
		u32 p2 = (u32) __umulhi(i, m);

		/* step 3 - compute qoutient*factor and subtract from P */
		u64 g  = q * f;
		u32 g2 = (u32) __umulhi(q, f);

		/* STEP 4: i = (p - g) with branchless code for correcting negatives */
		i  = p  - g;
		p2 = p2 - g2 - (i > p);  /* subtract carry if we had underflow in low part */

		if (p2) {
			i += f;
		}
	} while( --count ); // End of countdown loop

#endif

}
