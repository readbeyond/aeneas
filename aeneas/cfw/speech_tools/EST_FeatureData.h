/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                       Copyright (c) 1995,1996                         */
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
/*                                                                       */
/*                   Author :  Paul Taylor                               */
/* -------------------------------------------------------------------   */
/*                  EST_Date Class header file                           */
/*                                                                       */
/*************************************************************************/

#ifndef __EST_FeatureData_H__
#define __EST_FeatureData_H__

#include "EST_Val.h"
#include "EST_TVector.h"
#include "EST_TList.h"
#include "EST_rw_status.h"
#include "EST_types.h"
#include "EST_Features.h"


typedef EST_TMatrix<EST_Val> EST_ValMatrix;
typedef EST_TVector<EST_Val> EST_ValVector;

/** A class for containing arbitrary multi-dimensional feature data.

A number of fields are defined in the EST_FeatureData class, each of
which represents a measurable quantity, such as height, age or
gender. Any number of fields can be defined and each field can take a
float, integer or string value. The class holds multiple instances of
field values, representing samples taken from a population. 

Several statistical training, testing and analysis programs use
EST_FeatureData as input and output. Member functions exist for
comparing fields, extracting given named or numbered fields, can 
converting appropriate fields to a EST_FMatrix.

*/


class EST_FeatureSample : public EST_ValVector {
private:
    bool p_sub_fd;

    void default_vals();
    void free_internals();
    void alloc_internals();

    EST_Features *p_info;
    EST_FeatureSample &copy(const EST_FeatureSample &a);
public:
    EST_FeatureSample();
    EST_FeatureSample(const EST_FeatureSample &a);

    /**@name Information functions */
    //@{

    /** set number of samples to be held in object and allocate
    space for storing them */
/*
    int num_fields() const {return info().num_fields();}

    //@}

    EST_Val &a(int field)
    {return EST_ValVector::a(field);}

    const EST_Val &a(int field) const
    {return EST_ValVector::a(field);}

    EST_Val &a(const EST_String &name)
    {return EST_ValVector::a(info().field_index(name));}

    const EST_Val &a(const EST_String &name) const
    {return EST_ValVector::a(info().field_index(name));}

    /// const element access operator
//      const EST_Val &operator () (int sample, const EST_String &field);
    /// non-const element access operator
      EST_Val &operator () (const EST_String &field);
      EST_Val &operator () (int field);

    EST_FeatureSample &EST_FeatureSample::operator=
	(const EST_FeatureSample &f);

  friend ostream& operator << (ostream &st, const EST_FeatureSample &a);
  //@}
*/


};


class EST_FeatureData{
private:
    bool p_sub_fd;

    void default_vals();
    void free_internals();
    void alloc_internals();


    EST_FeatureData &copy(const EST_FeatureData &a);

    EST_Features info;
    EST_ValMatrix fd;
public:
    EST_FeatureData();
    EST_FeatureData(const EST_FeatureData &a);
    ~EST_FeatureData();
    EST_Features &ginfo() {return info;}

    int num_samples() const;
    int num_features() const;

    void resize(int num_samples, int num_columns, bool preserve = 1);
    void resize(int num_samples, EST_Features &f, bool preserve = 1);

    void set_num_samples(int num_samples, bool preserve=1);

/*    void extract_features(EST_FeatureData &f, const EST_StrList &fields) const;
    void extract_features(EST_FeatureData &f, EST_IList &fields) const;
*/

    EST_String type(const EST_String &feature_name);
    EST_StrList values(const EST_String &feature_name);

    void set_type(EST_String &feature_name, EST_String &type);
    void set_values(EST_String &feature_name, EST_StrList &values);

    int update_values(const EST_String &feature_name, int max);

    int feature_position(const EST_String &feature_name);


    EST_read_status load(const EST_String &name);

    EST_write_status save(const EST_String &name, 
			const EST_String &EST_filetype = "") const;

    EST_Val &a(int sample, int field);
    EST_Val &a(int sample, const EST_String &name);
    const EST_Val &a(int sample, int field) const;
    const EST_Val &a(int sample, const EST_String &name) const;

    friend ostream& operator << (ostream &st,const EST_FeatureData &a);

};


#endif /* __EST_FeatureData_H__ */
