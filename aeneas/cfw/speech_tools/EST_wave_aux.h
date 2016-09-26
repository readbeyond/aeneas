/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                         Copyright (c) 1996                            */
/*                        All Rights Reserved.                           */
/*                                                                       */
/*  Permission is hereby granted, free of charge, to use and distribute  */
/*  this software and its documentation without restriction, including   */
/*  without limitation the rights to use, copy, modify, merge, publish,  */
/*  distribute, sublicense, and/or sell copies of this work, and to      */
/*  permit persons to whom this work is furnished to do so, subject to   */
/*  the following conditions:                                            */
/*   1. The code must retain the above copyright notice, this list of    */
/*      conditions and the following disclaimer.                         */
/*   2. Any modifications must be clearly marked as such.                */
/*   3. Original authors' names are not deleted.                         */
/*   4. The authors' names are not used to endorse or promote products   */
/*      derived from this software without specific prior written        */
/*      permission.                                                      */
/*                                                                       */
/*  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK        */
/*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      */
/*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   */
/*  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE     */
/*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    */
/*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   */
/*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          */
/*  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       */
/*  THIS SOFTWARE.                                                       */
/*                                                                       */
/*************************************************************************/
/*             Author :  Paul Taylor and Alan Black                      */
/*             Date   :  May 1996                                        */
/*-----------------------------------------------------------------------*/
/*                EST_Wave class auxiliary functions                         */
/*                                                                       */
/*=======================================================================*/


/**@name EST_wave_aux
Auxiliary functions for processing waveforms.
  */

//@{

#ifndef __EST_WAVE_AUX_H__
#define __EST_WAVE_AUX_H__

#include "EST_String.h"
#include "EST_Wave.h"
#include "ling_class/EST_Relation.h"
#include "EST_Option.h"
#include "EST_FMatrix.h"
#include "EST_TNamedEnum.h"


int wave_extract_channel(EST_Wave &single, const EST_Wave &multi, int channel);

void wave_combine_channels(EST_Wave &combined, const EST_Wave &multi);

int wave_subwave(EST_Wave &subsig,EST_Wave &sig,int offset,int length);

int wave_divide(EST_WaveList &wl, EST_Wave &sig, EST_Relation &keylab,
		const EST_String &ext);

int wave_extract(EST_Wave &part, EST_Wave &sig, EST_Relation &keylab, 
		 const EST_String &file);

void add_waves(EST_Wave &s, const EST_Wave &m);

EST_Wave difference(EST_Wave &a, EST_Wave &b);
float rms_error(EST_Wave &a, EST_Wave &b, int channel);
float abs_error(EST_Wave &a, EST_Wave &b, int channel);
float correlation(EST_Wave &a, EST_Wave &b, int channel);

EST_FVector rms_error(EST_Wave &a, EST_Wave &b);
EST_FVector abs_error(EST_Wave &a, EST_Wave &b);
EST_FVector correlation(EST_Wave &a, EST_Wave &b);

EST_Wave error(EST_Wave &ref, EST_Wave &test, int relax);

void absolute(EST_Wave &a);

EST_read_status read_wave(EST_Wave &sig, const EST_String &in_file, 
			  EST_Option &al);
EST_write_status write_wave(EST_Wave &sig, const EST_String &in_file, EST_Option &al);
void wave_info(EST_Wave &w);
void invert(EST_Wave &sig);


void differentiate(EST_Wave &sig);
void reverse(EST_Wave &sig);

void ulaw_to_short(const unsigned char *ulaw,short *data,int length);
void alaw_to_short(const unsigned char *alaw,short *data,int length);
void uchar_to_short(const unsigned char *chars,short *data,int length);
void short_to_char(const short *data,unsigned char *chars,int length);
void short_to_ulaw(const short *data,unsigned char *ulaw,int length);
void short_to_alaw(const short *data,unsigned char *alaw,int length);

// Used when setting Waves in Features
VAL_REGISTER_CLASS_DCLS(wave,EST_Wave)

enum EST_sample_type_t {
  st_unknown, 
  st_schar, 
  st_uchar, 
  st_short,
  st_shorten, 
  st_int, 
  st_float,
  st_double,
  st_mulaw, 
  st_adpcm, 
  st_alaw, 
  st_ascii};


enum EST_write_status wave_io_save_header(FILE *fp,
                      const int num_samples, const int num_channels,
                      const int sample_rate,
                      const EST_String& stype, const int bo,
                      const EST_String& ftype);

extern EST_TNamedEnum<EST_sample_type_t> EST_sample_type_map;

#endif /* __EST_WAVE_AUX_H__ */

//@}
