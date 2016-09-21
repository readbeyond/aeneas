/*
 *	$Id: rateconv.cc,v 1.5 2014/04/07 15:32:10 robert Exp $
 *
 *	RATECONV.C
 *
 *	Convert sampling rate stdin to stdout
 *
 *	Copyright (c) 1992, 1995 by Markus Mummert
 *
 *****************************************************************************
 *      MODIFIED BY Alan W Black (awb@cstr.ed.ac.uk)
 *           Make it compilable under C++
 *           and integrate into Edinburgh Speech Tools (i.e. no longer
 *                reads from stdin / writes to stdout)
 *           Removed interface functions
 *           ansified function calls
 *           made it work in floats rather than ints
 *      I got the original from a random linux site, the original 
 *      author's email is  <mum@mmk.e-technik.tu-muenchen.de>
 *****************************************************************************
 *
 *	Redistribution and use of this software, modification and inclusion
 *	into other forms of software are permitted provided that the following
 *	conditions are met:
 *
 *	1. Redistributions of this software must retain the above copyright
 *	   notice, this list of conditions and the following disclaimer.
 *	2. If this software is redistributed in a modified condition
 *	   it must reveal clearly that it has been modified.
 *	
 *	THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS''
 *	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 *	TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 *	PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR
 *	CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *	EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *	PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *	PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *	OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 *	USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 *	DAMAGE.
 *
 *
 *	history: 2.9.92		begin coding
 *		 5.9.92		fully operational
 *		 14.2.95 	provide BIG_ENDIAN, SWAPPED_BYTES_DEFAULT
 *				switches, Copyright note and References
 *		 25.11.95	changed XXX_ENDIAN to I_AM_XXX_ENDIAN
 *				default gain set to 0.8
 *		 3.12.95	stereo implementation
 *				SWAPPED_BYTES_DEFAULT now HBYTE1ST_DEFAULT
 *				changed [L/2] to (L-1)/2 for exact symmetry
 *
 *
 *	IMPLEMENTATION NOTES
 *
 *	Converting is achieved by interpolating the input samples in
 *	order to evaluate the represented continuous input slope at
 *	sample instances of the new rate (resampling). It is implemented 
 *	as a polyphase FIR-filtering process (see reference). The rate
 *	conversion factor is determined by a rational factor. Its
 *	nominator and denominator are integers of almost arbitrary
 *	value, limited only by coefficient memory size.
 *
 *	General rate conversion formula:
 *
 *	    out(n*Tout) = SUM in(m*Tin) * g((n*d/u-m)*Tin) * Tin
 *		      over all m
 *
 *	FIR-based rate conversion formula for polyphase processing:
 *
 *			  L-1
 *	    out(n*Tout) = SUM in(A(i,n)*Tin) * g(B(i,n)*Tin) * Tin
 *			  i=0
 *
 *	    A(i,n) = i - (L-1)/2 + [n*d/u]              
 *	           = i - (L-1)/2 + [(n%u)*d/u] + [n/u]*d 
 *	    B(i,n) = n*d/u - [n*d/u] + (L-1)/2 - i
 *	           =  ((n%u)*d/u)%1  + (L-1)/2 - i
 *	    Tout   = Tin * d/u
 *
 *	  where:
 *	    n,i		running integers
 *	    out(t)	output signal sampled at t=n*Tout
 *	    in(t)	input signal sampled in intervals Tin
 *	    u,d		up- and downsampling factor, integers
 *	    g(t)	interpolating function
 *	    L		FIR-length of realized g(t), integer
 *	    /		float-division-operator
 *	    %		float-modulo-operator
 *	    []		integer-operator
 *
 *	  note:
 *	    (L-1)/2	in A(i,n) can be omitted as pure time shift yielding
 *			a causal design with a delay of ((L-1)/2)*Tin.
 *	    n%u		is a cyclic modulo-u counter clocked by out-rate
 *	    [n/u]*d	is a d-increment counter, advanced when n%u resets
 *	    B(i,n)*Tin	can take on L*u different values, at which g(t)
 *			has to be sampled as a coefficient array
 *
 *	Interpolation function design:
 *
 * 	    The interpolation function design is based on a sinc-function
 *	    windowed by a gaussian function. The former determines the
 *	    cutoff frequency, the latter limits the necessary FIR-length by
 *	    pushing the outer skirts of the resulting impulse response below
 *	    a certain threshold fast enough. The drawback is a smoothed
 *	    cutoff inducing some aliasing. Due to the symmetry of g(t) the
 *	    group delay of the filtering process is constant (linear phase).
 *
 *	    g(t) = 2*fgK*sinc(pi*2*fgK*t) * exp(-pi*(2*fgG*t)**2)
 *
 *	  where:
 *	    fgK		cutoff frequency of sinc function in f-domain
 *	    fgG		key frequency of gaussian window in f-domain
 *			reflecting the 6.82dB-down point
 *
 * 	  note:	    
 *	    Taking fsin=1/Tin as the input sampling frequency, it turns out
 *	    that in conjunction with L, u and d only the ratios fgK/(fsin/2)
 *	    and fgG/(fsin/2) specify the whole process. Requiring fsin, fgK
 *	    and fgG as input purposely keeps the notion of absolute
 *	    frequencies.
 *
 *	Numerical design:
 *
 *	    Samples are expected to be 16bit-signed integers, alternating
 *	    left and right channel in case of stereo mode- The byte order
 *	    per sample is selectable. FIR-filtering is implemented using
 *	    32bit-signed integer arithmetic. Coefficients are scaled to
 *	    find the output sample in the high word of accumulated FIR-sum.
 *
 *	    Interpolation can lead to sample magnitudes exceeding the
 *	    input maximum. Worst case is a full scale step function on the
 *	    input. In this case the sinc-function exhibits an overshoot of
 *	    2*9=18percent (Gibb's phenomenon). In any case sample overflow
 *	    can be avoided by a gain of 0.8.
 *
 *	    If u=d=1 and if the input stream contains only a single sample,
 *	    the whole length of the FIR-filter will be written to the output.
 *	    In general the resulting output signal will be (L-1)*fsout/fsin
 *	    samples longer than the input signal. The effect is that a 
 *	    finite input sequence is viewed as padded with zeros before the
 *	    `beginning' and after the `end'. 
 *
 *	    The output lags ((L-1)/2)*Tin behind to implement g(t) as a
 *	    causal system corresponding to a causal relationship of the
 *	    discrete-time sequences in(m/fsin) and out(n/fsout) with
 *	    respect to a sequence time origin at t=n*Tin=m*Tout=0.
 *
 *
 * 	REFERENCES
 *
 *	    Crochiere, R. E., Rabiner, L. R.: "Multirate Digital Signal
 *	    Processing", Prentice-Hall, Englewood Cliffs, New Jersey, 1983
 *
 *	    Zwicker, E., Fastl, H.: "Psychoacoustics - Facts and Models",
 *	    Springer-Verlag, Berlin, Heidelberg, New-York, Tokyo, 1990
 */

#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <cstring>
#include "rateconv.h"

/*
 *	adaptable defines and globals
 */
#define	BYTE		char		/* signed or unsigned */
#define	WORD		short		/* signed or unsigned, fit two BYTEs */
#define	LONG 		int		/* signed, fit two WORDs */

#ifndef MAXUP
#define	MAXUP		0x400		/* MAXUP*MAXLENGTH worst case malloc */
#endif

#ifndef MAXLENGTH
#define	MAXLENGTH	0x400		/* max FIR length */
#endif
					/* accounts for mono samples, means */
#define OUTBUFFSIZE 	(2*MAXLENGTH)	/* fit >=MAXLENGHT stereo samples */
#define INBUFFSIZE	(4*MAXLENGTH)	/* fit >=2*MAXLENGTH stereo samples */
#define sqr(a)	((a)*(a))

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

/* AWB deleted previous byte swap globals, byteswap is done external to */
/* this function                                                        */

#ifdef	STEREO_DEFAULT
static  int	g_monoflag = 0;
#else
static  int	g_monoflag = -1;
#endif

/*
 *	other globals
 */
static double	g_ampli = 0.8;			/* default gain, don't change */
static int
/*	g_infilehandle = 0,	*/	/* stdin */
/*	g_outfilehandle = 1,	*/	/* stdout */
	g_firlen,			/* FIR-length */
	g_up,				/* upsampling factor */
	g_down				/* downsampling factor */
;

static float
	g_sin[INBUFFSIZE],		/* input buffer */
	g_sout[OUTBUFFSIZE],		/* output buffer */
	*g_coep;			/* coefficient array pointer */

static double
	g_fsi,				/* input sampling frequency */
	g_fgk,				/* sinc-filter cutoff frequency */
	g_fgg				/* gaussian window key frequency */
;					/* (6.8dB down freq. in f-domain) */
	

/*
 *	evaluate sinc(x) = sin(x)/x safely
 */
static double sinc(double x)
{
    return(fabs(x) < 1E-50 ? 1.0 : sin(fmod(x,2*M_PI))/x);
}

/*
 *	evaluate interpolation function g(t) at t
 *	integral of g(t) over all times is expected to be one
 */
static double interpol_func(double t,double fgk,double fgg)
{
    return (2*fgk*sinc(M_PI*2*fgk*t)*exp(-M_PI*sqr(2*fgg*t)));
}

/*
 *	evaluate coefficient from i, q=n%u by sampling interpolation function 
 *	and scale it for integer multiplication used by FIR-filtering
 */
static float coefficient(int i,int q,int firlen,double fgk,double fgg,
			 double fsi,int up,int down,double amp)
{
    float val;
    double d;

    d = interpol_func((fmod(q*down/(double)up,1.) + (firlen-1)/2. - i) / fsi,
		      fgk,
		      fgg);
    val =  amp * d/fsi;
    return val;
}

/*
 *	transfer n floats from  s to d
 */
static void transfer_int(float *s,float *d,int n)
{
    memmove(d,s,sizeof(float)*n);
}

/*
 *	zerofill n floats from s 
 */
static void zerofill(float *s,int n)
{
    memset(s,0,n*(sizeof(float)));
}

/*
 *	FIR-routines, mono and stereo
 *	this is where we need all the MIPS
 */
void fir_mono(float *inp,float *coep,int firlen,float *outp)
{
    float akku = 0, *endp;
    int n1 = (firlen / 8) * 8, n0 = firlen % 8;

    endp = coep + n1;
    while (coep != endp) {
	akku += *inp++ * *coep++;
	akku += *inp++ * *coep++;
	akku += *inp++ * *coep++;
	akku += *inp++ * *coep++;
	akku += *inp++ * *coep++;
	akku += *inp++ * *coep++;
	akku += *inp++ * *coep++;
	akku += *inp++ * *coep++;
    }

    endp = coep + n0;
    while (coep != endp) {
	akku += *inp++ * *coep++;
    }
    
    *outp = akku;
}

static void fir_stereo(float *inp,float *coep,int firlen,float *out1p,float *out2p)
{
    float akku1 = 0, akku2 = 0, *endp;
    int n1 = (firlen / 8) * 8, n0 = firlen % 8;

    endp = coep + n1;
    while (coep != endp) {
	akku1 += *inp++ * *coep;
	akku2 += *inp++ * *coep++;
	akku1 += *inp++ * *coep;
	akku2 += *inp++ * *coep++;
	akku1 += *inp++ * *coep;
	akku2 += *inp++ * *coep++;
	akku1 += *inp++ * *coep;
	akku2 += *inp++ * *coep++;
	akku1 += *inp++ * *coep;
	akku2 += *inp++ * *coep++;
	akku1 += *inp++ * *coep;
	akku2 += *inp++ * *coep++;
	akku1 += *inp++ * *coep;
	akku2 += *inp++ * *coep++;
	akku1 += *inp++ * *coep;
	akku2 += *inp++ * *coep++;
    }

    endp = coep + n0;
    while (coep != endp) {
	akku1 += *inp++ * *coep;
	akku2 += *inp++ * *coep++;
    }
    *out1p = akku1;
    *out2p = akku2;
}

/*
 * 	filtering from input buffer to output buffer;
 *	returns number of processed samples in output buffer:
 *	if it is not equal to output buffer size,
 *	the input buffer is expected to be refilled upon entry, so that
 *	the last firlen numbers of the old input buffer are
 *	the first firlen numbers of the new input buffer;
 *	if it is equal to output buffer size, the output buffer
 *	is full and is expected to be stowed away;
 *
 */
static int inbaseidx = 0, inoffset = 0, cycctr = 0, outidx = 0;

static int filtering_on_buffers
	(float *inp,int insize,float *outp, int outsize, 
	 float *coep,int firlen,int up,int down,int monoflag)
{

    if (monoflag) {
	while (-1) {
	    inoffset = (cycctr * down)/up;
	    if ((inbaseidx + inoffset + firlen) > insize) {
		inbaseidx -= insize - firlen + 1;
		return(outidx);
	    }
	    fir_mono(inp + inoffset + inbaseidx,
		     coep + cycctr * firlen,
		     firlen, outp + outidx++);
	    cycctr++;
	    if (!(cycctr %= up))
		inbaseidx += down;
	    if (!(outidx %= outsize))
		return(outsize);
	}
    } 
    else {
	/*
	 * rule how to convert mono routine to stereo routine:
	 * firlen, up, down and cycctr relate to samples in general,
	 * wether mono or stereo; inbaseidx, inoffset and outidx as
	 * well as insize and outsize still account for mono samples.
	 */
	while (-1) {
	    inoffset = 2*((cycctr * down)/up);
	    if ((inbaseidx + inoffset + 2*firlen) > insize) {
		inbaseidx -= insize - 2*firlen + 2;
		return(outidx);
	    }
/* order?
			   fir_stereo(inp + inoffset + inbaseidx,
		       coep + cycctr * firlen, firlen,
		       outp + outidx++, outp + outidx++);

*/ 
	    fir_stereo(inp + inoffset + inbaseidx,
		       coep + cycctr * firlen, firlen,
		       outp + outidx, outp + outidx+1);
	    outidx += 2;

		cycctr++;
	    if (!(cycctr %= up))
		inbaseidx += 2*down;
	    if (!(outidx %= outsize))
		return(outsize);
	}
    }
}

/*
 *	set up coefficient array
 */
static void make_coe(void)
{
	int i, q;

	for (i = 0; i < g_firlen; i++) {
	    for (q = 0; q < g_up; q++) {
		g_coep[q * g_firlen + i] = coefficient(i, q, g_firlen,
		    g_fgk, g_fgg, g_fsi, g_up, g_down, g_ampli);
	    }
	}
}

/***********************************************************************/
/*  Serious modifications by Alan W Black (awb@cstr.ed.ac.uk)          */
/*  to interface with rest of system // deleted various io functions   */
/*  too.                                                               */
/***********************************************************************/
static WORD *inbuff = NULL;
static int inpos;
static int inmax;
static WORD *outbuff = NULL;
static int outpos;
static int outmax;

static int ioerr(void)
{
    delete[] g_coep;
    return -1;
}

static int gcd(int x, int y)
{
    int remainder,a,b;

    if ((x < 1) || (y < 1))
	return -1;

    for (a=x,b=y; b != 0; )
    {
	remainder = a % b;
	a = b;
	b = remainder; 
    }
    return a;
}

static int find_ratios(int in_samp_freq,int out_samp_freq,int *up,int *down)
{
    // Find ratios
    int d;

    d = gcd(in_samp_freq,out_samp_freq);
    if (d == -1) return -1;
    *down = in_samp_freq / d;
    *up = out_samp_freq / d;

    if ((*up > 1024) || (*down > 1024))
	return -1;   // should try harder

    return 0;
}

static int intimport(float *buff, int n)
{
    /* Import n more samples from PWave into buff */
    int i,end;

    if ((inpos+n) >= inmax)
	end = inmax - inpos;
    else
	end = n;
    for (i=0;i < end; i++)
	buff[i] = inbuff[inpos++];

    return i;
}
    
static int intexport(float *buff, int n)
{
    /* Export n samples from buff into end of PWave */
    int i,end;

    if ((outpos+n) >= outmax)
	end = outmax - inpos;
    else
	end = n;
    for (i=0;i < end; i++)
	outbuff[outpos++] = (short)buff[i];

    return i;
}
    
static int init_globs(WORD *in,int insize, WORD **out, int *outsize,
		       int in_samp_freq, int out_samp_freq)
{
    int new_size;
    g_monoflag = 1;		/* always mono */
    if (find_ratios(in_samp_freq,out_samp_freq,&g_up,&g_down) == -1)
	return -1;
    g_fsi = 1.0; /* ? in_samp_freq ? */
    if (g_up > g_down)
    {   // upsampling
	g_fgg = 0.0116;
	g_fgk = 0.461;
	g_firlen = (int)(162 * (float)g_up/(float)g_down);
    }
    else
    {   // downsampling
	g_fgg = (float)g_up/(float)g_down * 0.0116;
	g_fgk = (float)g_up/(float)g_down * 0.461;
	g_firlen = (int)(162 * (float)g_down/(float)g_up);
    }
    if (g_firlen < 1 || g_firlen > MAXLENGTH)
	return -1;
    g_ampli = 0.8;
    g_coep = new float[g_firlen * g_up];

    inpos = 0;
    inmax = insize;
    inbuff = in;
    new_size = (int)(((float)out_samp_freq/(float)in_samp_freq)*
		     1.1*insize)+2000;
    *out = new WORD[new_size];
    outbuff = *out;
    outmax = new_size;
    *outsize = 0;
    outpos = 0;

    /* For filter_on_buffers */
    inbaseidx = 0;
    inoffset = 0;
    cycctr = 0;
    outidx = 0;

    return 0;
}


/*
 * External call added by Alan W Black, 4th June 1996
 * a combination of parse args and main
 */
int rateconv(short *in,int isize, short **out, int *osize,
	     int in_samp_freq, int out_samp_freq)
{
    int insize = 0, outsize = 0, skirtlen;

    if (init_globs(in,isize,out,osize,in_samp_freq,out_samp_freq) == -1)
	return -1;

    make_coe();
    skirtlen = (g_firlen - 1) * (g_monoflag ? 1 : 2);
    zerofill(g_sin, skirtlen);
    do {
	insize = intimport(g_sin + skirtlen, INBUFFSIZE - skirtlen);
	if (insize < 0 || insize > INBUFFSIZE - skirtlen) 
	    return ioerr();
	do {
	    outsize = filtering_on_buffers(g_sin, skirtlen + insize,
					   g_sout, OUTBUFFSIZE, 
					   g_coep, g_firlen, g_up, g_down,
					   g_monoflag);
	    if (outsize != OUTBUFFSIZE) {
		transfer_int(g_sin + insize, g_sin, skirtlen);
		break;
	    }
	    if (intexport(g_sout, outsize) != outsize) 
		return ioerr();
	} while (-1);
    } while (insize > 0);
    zerofill(g_sin + skirtlen, skirtlen);
    do {
	outsize = filtering_on_buffers(g_sin, skirtlen + skirtlen,
				       g_sout, OUTBUFFSIZE, 
				       g_coep, g_firlen, g_up, g_down,
				       g_monoflag);
	if (intexport(g_sout, outsize) != outsize) 
	    return ioerr();
    } while (outsize == OUTBUFFSIZE); 

    delete[] g_coep;

    *osize = outpos;

    /* The new signal will be offset by half firlen window so fix it */
    memmove(*out,*out+g_firlen/4,*osize*2);
    *osize -= g_firlen/4;

    return 0;

}

