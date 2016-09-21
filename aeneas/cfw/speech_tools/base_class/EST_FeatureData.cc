/************************************************************************/
/*                                                                      */
/*                Centre for Speech Technology Research                 */
/*                     University of Edinburgh, UK                      */
/*                       Copyright (c) 1996,1997                        */
/*                        All Rights Reserved.                          */
/*                                                                      */
/*  Permission is hereby granted, free of charge, to use and distribute */
/*  this software and its documentation without restriction, including  */
/*  without limitation the rights to use, copy, modify, merge, publish, */
/*  distribute, sublicense, and/or sell copies of this work, and to     */
/*  permit persons to whom this work is furnished to do so, subject to  */
/*  the following conditions:                                           */
/*   1. The code must retain the above copyright notice, this list of   */
/*      conditions and the following disclaimer.                        */
/*   2. Any modifications must be clearly marked as such.               */
/*   3. Original authors' names are not deleted.                        */
/*   4. The authors' names are not used to endorse or promote products  */
/*      derived from this software without specific prior written       */
/*      permission.                                                     */
/*                                                                      */
/*  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK       */
/*  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING     */
/*  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT  */
/*  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE    */
/*  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES   */
/*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN  */
/*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,         */
/*  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF      */
/*  THIS SOFTWARE.                                                      */
/*                                                                      */
/************************************************************************/
/*                                                                      */
/*                   Author: Paul Taylor Caley                          */
/*                       Date: July 1998                                */
/* -------------------------------------------------------------------- */
/*                     Feature Data Class                               */
/*                                                                      */
/************************************************************************/

#include "EST_TMatrix.h"
#include "EST_Val.h"
#include "EST_FeatureData.h"
#include "EST_string_aux.h"
#include "EST_Token.h"
#include "EST_FileType.h"
#include "EST_error.h"
#include <iostream>
#include <fstream>

#include "EST_THash.h"


EST_FeatureData::EST_FeatureData()
{
    default_vals();
}



EST_FeatureData::EST_FeatureData(const EST_FeatureData &a)
{ 
    default_vals();
    copy(a);
}

EST_FeatureData::~EST_FeatureData(void)
{
}

int EST_FeatureData::num_samples() const
{
    return fd.num_rows();
}

int EST_FeatureData::num_features() const
{
    return fd.num_columns();
}


void EST_FeatureData::default_vals()
{
/*    cout << "Default values\n";
    p_sub_fd = false;
    p_info = new EST_FeatureInfo;
*/
}

void EST_FeatureData::set_num_samples(int num_samples, bool preserve)
{
    fd.resize(num_samples, fd.num_columns(), preserve);
}

void EST_FeatureData::resize(int num_samples, int num_features, bool preserve)
{
    // If enlargement is required, give new features dummy names
    // and set their types to <STRING>. If preserve is set to 0
    // rename all features this way.

    if (num_features > fd.num_columns())
    {
	int i;
	if (preserve)
	    i = fd.num_columns();
	else
	    i = 0;
	for (; i < num_features; ++i)
	    info.set("unnamed_" + itoString(i), "<STRING>");
    }

    fd.resize(num_samples, num_features, preserve);
}

void EST_FeatureData::resize(int num_samples, EST_Features &f, bool preserve)
{
    fd.resize(num_samples, f.length(), preserve);
    info = f;
}

EST_String EST_FeatureData::type(const EST_String &feature_name)
{
    EST_String t = info.S(feature_name);
    
    if (t.contains("<", 0)) // i.e. a predefined type
	return t;

    return "undef";
}

EST_StrList EST_FeatureData::values(const EST_String &feature_name)
{
    EST_StrList v;
    EST_String t = info.S(feature_name);
    
    // check for infinite set:
    if ((t == "<FLOAT>") || (t == "<INT>") || (t == "<STRING>"))
	return v; 

    StringtoStrList(t, v);
    return v;
}

int EST_FeatureData::feature_position(const EST_String &feature_name)
{
    int i;

    EST_Features::Entries p;

    for (i = 0, p.begin(info); p; ++p, ++i)
    {
//	cout << "looking at " << info.fname(p) << endl;
//	cout << "i = " << i << endl;
	if (p->k == feature_name)
	    return i;
    }

    EST_error("No such feature %s\n", (const char *) feature_name);
    return 0;
}

int EST_FeatureData::update_values(const EST_String &feature_name, int max)
{
    // This should be converted back to Hash tables once extra
    // iteration functions are added the EST_Hash.
    int i, col;
    EST_Features values;
    EST_String v;

//    EST_TStringHash<int> values(max);

    col = feature_position(feature_name);

    for (i = 0; i < num_samples(); ++i)
	values.set(fd.a(i, col).string(), 1);

    // check to see if there are more types than allowed, if so
    // just set to open set STRING
    if (values.length() > max)
	v = "<STRING>"; 
    else
      {
	EST_Features::Entries p;
	for(p.begin(values); p; ++p)
	    v += p->k + " ";
      }

    info.set(feature_name, v);
	
    return values.length();
}

EST_FeatureData & EST_FeatureData::copy(const EST_FeatureData &a)
{
    (void) a;
/*    // copy on a sub can't alter header information
    if (!p_sub_fd)
    {
	delete p_info;
	*p_info = *(a.p_info);
    }
    // but data can be copied so long as no resizing is involved.
    EST_ValMatrix::operator=(a);
*/
    return *this;
}

/*void EST_FeatureData::a(int i, int j)
{
    return EST_ValMatrix::a(i, j);
}
*/
/*
EST_Val &EST_FeatureData::operator()(int i, int j)
{
    return a(i, j);
}

EST_Val &EST_FeatureData::operator()(int s, const EST_String &f)
{
    int i = info().field_index(f);
    return a(s, i);
}

EST_FeatureData &EST_FeatureData::operator=(const EST_FeatureData &f)
{
    return copy(f);
}

*/
EST_Val &EST_FeatureData::a(int i, const EST_String &f)
{
  (void)f;
  return fd.a(i, 0);
}

EST_Val &EST_FeatureData::a(int i, int j)
{
    return fd.a(i, j);
}
const EST_Val &EST_FeatureData::a(int i, const EST_String &f) const
{
  (void)f;
    return fd.a(i, 0);
}

const EST_Val &EST_FeatureData::a(int i, int j) const
{
    return fd.a(i, j);
}


/*
void EST_FeatureData::sub_samples(EST_FeatureData &f, int start, int num)
{
    sub_matrix(f, start, num);
    f.p_info = p_info;
    f.p_sub_fd = true;
}

void EST_FeatureData::extract_named_fields(const EST_String &fields)
{
    EST_FeatureData n;
    // there must be a more efficient way than a copy?
    extract_named_fields(n, fields);
    *this = n;
}

void EST_FeatureData::extract_named_fields(const EST_StrList &fields)
{
    EST_FeatureData n;
    // there must be a more efficient way than a copy?
    extract_named_fields(n, fields);
    *this = n;
}

void EST_FeatureData::extract_numbered_fields(const EST_String &fields)
{
    EST_FeatureData n;
    // there must be a more efficient way than a copy?
    extract_numbered_fields(n, fields);
    *this = n;
}

void EST_FeatureData::extract_numbered_fields(const EST_IList &fields)
{
    EST_FeatureData n;
    // there must be a more efficient way than a copy?
    extract_numbered_fields(n, fields);
    *this = n;
}


void EST_FeatureData::extract_named_fields(EST_FeatureData &f, 
					   const EST_String &fields) const
{
    EST_StrList s;

    StringtoStrList(fields, s);
    extract_named_fields(f, s);
}
void EST_FeatureData::extract_named_fields(EST_FeatureData &f, 
					   const EST_StrList &n_fields) const
{
    EST_Litem *p;
    EST_StrList n_types;
    int i, j;

    info().extract_named_fields(*(f.p_info), n_fields);

    for (p = n_fields.head(), i = 0; i < f.num_fields(); ++i, p = p->next())
	for (j = 0; j < f.num_samples(); ++j)
	    f(j, i) = a(j, n_fields(p));

}

void EST_FeatureData::extract_numbered_fields(EST_FeatureData &f, 
					      const EST_IList &fields) const
{
    EST_Litem *p;
    EST_StrList n_fields;
    int i, j;

    for (p = fields.head(); p; p = p->next())
	n_fields.append(info().field_name(fields(p)));
    
    info().extract_named_fields(*(f.p_info), n_fields);

    for (p = fields.head(), i = 0; i < f.num_fields(); ++i, p = p->next())
	for (j = 0; j < f.num_samples(); ++j)
	    f(j, i) = a(j, fields(p));

}

void EST_FeatureData::extract_numbered_fields(EST_FeatureData &f, 
					      const EST_String &fields) const
{
    EST_StrList s;
    EST_IList il;

    StringtoStrList(fields, s);
    StrListtoIList(s, il);
    extract_numbered_fields(f, il);
}
*/

EST_write_status save_est(const EST_FeatureData &f, const EST_String &filename)
{
  (void)f;
  (void)filename;
/*    
    ostream *outf;
    EST_Litem *s, *e;
    int i;
    if (filename == "-")
	outf = &cout;
    else
	outf = new ofstream(filename);
    
    if (!(*outf))
	return write_fail;
    
    outf->precision(5);
    outf->setf(ios::fixed, ios::floatfield);
    outf->width(8);
    
    *outf << "EST_File feature_data\n"; // EST header identifier
    *outf << "DataType ascii\n";
    *outf << "NumSamples " << f.num_samples() << endl;
    *outf << "NumFields " << f.num_fields() << endl;
    *outf << "FieldNames " << f.info().field_names();
    *outf << "FieldTypes " << f.info().field_types();
    if (f.info().group_start.length() > 0)
	for (s = f.info().group_start.head(), e = f.info().group_end.head(); 
	     s; s = s->next(), e = e->next())
	    *outf << "Group " << f.info().group_start.key(s) << " " << 
		f.info().group_start.val(s) << " " << f.info().group_end.val(e) << endl;

    for (i = 0; i < f.num_fields(); ++i)
	if (f.info().field_values(i).length() > 0)
	    *outf << "Field_" << i << "_Values " 
		<< f.info().field_values(i) << endl;

    *outf << "EST_Header_End\n"; // EST end of header identifier

//    *outf << ((EST_ValMatrix ) f);
    *outf << f;
    */

    return write_ok;
}


EST_write_status EST_FeatureData::save(const EST_String &filename, 
				       const EST_String &file_type) const
{
    if ((file_type == "est") || (file_type == ""))
	return save_est(*this, filename);
/*    else if (file_type = "octave")
	return save_octave(*this, filename);
    else if (file_type = "ascii")
	return save_ascii(*this, filename);
*/

    cerr << "Can't save feature data in format \"" << file_type << endl;
    return write_fail;
}



EST_read_status EST_FeatureData::load(const EST_String &filename)
{
    int i, j;
    EST_Option hinfo;
    EST_String k, v;
    EST_read_status r;
    bool ascii;
    EST_TokenStream ts;
    EST_EstFileType t;
    int ns, nf;

    if (((filename == "-") ? ts.open(cin) : ts.open(filename)) != 0)
    {
	cerr << "Can't open track file " << filename << endl;
	return misc_read_error;
    }
    // set up the character constant values for this stream
    ts.set_SingleCharSymbols(";");
    ts.set_quotes('"','\\');

    if ((r = read_est_header(ts, hinfo, ascii, t)) != format_ok)
    {
	cerr << "Error reading est header of file " << filename << endl;
	return r;
    }

    if (t != est_file_feature_data)
    {
	cerr << "Not a EST Feature Data file: " << filename << endl;
	return misc_read_error;
    }

    ns = hinfo.ival("NumSamples");
    nf = hinfo.ival("NumFeatures");
    
    cout << "ns: " << ns << endl;
    cout << "nf: " << nf << endl;
    resize(ns, nf);

    info.clear(); // because resize will make default names

    for (i = 0; i < nf; ++i)
    {
	k = "Feature_" + itoString(i+1);
	if (hinfo.present(k))
	{
	    v = hinfo.val(k);
	    info.set(v.before(" "), v.after(" "));
	    cout << "value: " << v.after(" ") << endl;
	}
	else
	    EST_error("No feature definition given for feature %d\n", i);
    }

    for (i = 0; i < ns; ++i)
      {
	EST_Features::Entries p;
	for (p.begin(info), j = 0; j < nf; ++j, ++p)
	{
	    if (p->k == "<FLOAT>")
	      a(i, j) = atof(ts.get().string());
	    else if (p->k == "<BOOL>")
		a(i, j) = atoi(ts.get().string());
	    else if (p->k == "<INT>")
		a(i, j) = atoi(ts.get().string());
	    else
		a(i, j) = ts.get().string();
	}
      }

    return format_ok;
}

/*ostream& operator << (ostream &st, const EST_FeatureInfo &a)
{   

//    st << a.field_names() << endl;
//    st << a.field_types() << endl;

    return st;
}
*/

ostream& operator << (ostream &st, const EST_FeatureData &d)
{   
    int i, j;
    EST_String t;
    EST_Val v;

//    st << a;

//    EST_ValMatrix::operator<<(st, (EST_ValMatrix)a);

    for (i = 0; i < d.num_samples(); ++i)
    {
	for (j = 0; j < d.num_features(); ++j)
	{
	    v =  d.a(i, j);
	    st << v  << " ";
//	    cout << "field type " << a.info().field_type(j) << endl;
/*	    else if (a.info().field_type(j) == "float")
		st << a.a(i, j);
	    else if (a.info().field_type(j) == "int")
		st << a.a(i, j);

	    else if (a.info().field_type(j) == "string")
	    {
		//		st << "\"" << a.a(i, j) << "\"";
		t = a.a(i, j);
		t.gsub(" ", "_");
		st << t;
	    }
*/
	}
	st << endl;
    }

    return st;
}
