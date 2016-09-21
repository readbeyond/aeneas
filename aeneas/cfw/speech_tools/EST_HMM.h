/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                      Copyright (c) 1995,1996                          */
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
/*             Author :  Paul Taylor                                     */
/*             Date   :  June 1996                                       */
/*-----------------------------------------------------------------------*/
/*             HMM Class header file                                     */
/*                                                                       */
/*=======================================================================*/

#ifndef __EST_HMM_H__
#define __EST_HMM_H__

#include "EST_String.h"
#include "EST_FMatrix.h"
#include "EST_Token.h"

class HMM_Mixture {
private:
public:
    HMM_Mixture();
    HMM_Mixture(const HMM_Mixture &s);
    HMM_Mixture(int n);

    void init(int vsize);

    EST_FVector mean;
    EST_FVector var;
    float gconst;
    int vecsize;
    HMM_Mixture& operator = (const HMM_Mixture& a);
};

ostream& operator<<(ostream& s, const HMM_Mixture &mix);

class HMM_State {
private:
public:
    HMM_State();
    HMM_State(const HMM_State &s);
    HMM_State(int n);

    void init(int n_mixes, int vsize);

    EST_TVector<HMM_Mixture> mixture;
    EST_FVector m_weight;

    HMM_State& operator = (const HMM_State& a);
};

ostream& operator<<(ostream& s, const HMM_State &st);

class HMM {
private:
public:
    HMM();
    HMM(int n, int v);

    void init(int n_states, int vsize, int n_streams);

    EST_String name;

    void clear();

    EST_String covkind;
    EST_String durkind;
    EST_String sampkind;

    EST_TVector<HMM_State> state;
    EST_FMatrix trans;

    int num_streams;
    int vecsize;
    
    EST_read_status load(EST_String file);
    EST_read_status load_portion(EST_TokenStream &ts, int v_size, 
				 int n_streams);
    EST_write_status save(EST_String file);
    HMM& operator = (const HMM& a);

    void balls(void);
};



ostream& operator<<(ostream& s, const HMM &model);

int operator !=(const HMM_Mixture &a, const HMM_Mixture &b);
int operator !=(const HMM_State &a, const HMM_State &b);


typedef EST_TList<HMM> EST_HMMList;



#endif /* __EST_HMM_H__ */
