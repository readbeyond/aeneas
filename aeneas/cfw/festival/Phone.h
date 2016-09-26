/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                       Copyright (c) 1996,1997                         */
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
/*             Author :  Alan W Black                                    */
/*             Date   :  April 1996                                      */
/*-----------------------------------------------------------------------*/
/*               Phone and PhoneSet class header file                    */
/*                                                                       */
/*=======================================================================*/
#ifndef __PHONE_H__
#define __PHONE_H__

class Phone{
 private:
    EST_String name;
    EST_StrStr_KVL features;
public:
    Phone() {name = "";}
    EST_String &phone_name() {return name;}
    void set_phone_name(const EST_String &p) {name = p;}
    void add_feat(const EST_String &f, const EST_String &v) 
          { features.add_item(f,v); }
    const EST_String &val(const EST_String &key) const 
       { return features.val_def(key,"");}
    const EST_String &val(const EST_String &key,const EST_String &def)
       { return features.val_def(key,def); }
    int match_features(Phone *foreign);

    inline friend ostream& operator<<(ostream& s, Phone &p);

    Phone & operator =(const Phone &a);
};

inline ostream& operator<<(ostream& s, Phone &p)
{
    s << "[PHONE " << p.phone_name() << "]";
//    s << p.features << endl;
    return s;
}

inline Phone &Phone::operator = (const Phone &a)
{
    name = a.name;
    features = a.features;
    return *this;
}


class PhoneSet{
 private:
    EST_String psetname;
    LISP silences;
    LISP map;
    LISP feature_defs;  // List of features and values 
    LISP phones;
public:
    PhoneSet() {psetname = ""; phones=feature_defs=map=silences=NIL;
	        gc_protect(&silences); gc_protect(&map);
                gc_protect(&feature_defs); gc_protect(&phones);}
    ~PhoneSet();
    const EST_String &phone_set_name() const {return psetname;}
    void set_phone_set_name(const EST_String &p) {psetname = p;}
    int present(const EST_String &phone) const 
       {return (siod_assoc_str(phone,phones) != NIL);}
    int is_silence(const EST_String &ph) const;
    void set_silences(LISP sils);
    void set_map(LISP m);
    LISP get_silences(void) {return silences;}
    LISP get_phones(void) {return phones;}
    LISP get_feature_defs(void) {return reverse(feature_defs);}
    int num_phones(void) const {return siod_llength(phones);}
    Phone *member(const EST_String &phone) const;
    int phnum(const char *phone) const;
    const char *phnum(const int n) const;
    int add_phone(Phone *phone);
    int feat_val(const EST_String &feat, const EST_String &val)
       { return (siod_member_str(val,
				 car(cdr(siod_assoc_str(feat,feature_defs))))
		           != NIL); }
    void set_feature(const EST_String &name, LISP vals); 

    inline friend ostream& operator<<(ostream& s, PhoneSet &p);

    Phone *find_matched_phone(Phone *phone);
    PhoneSet & operator =(const PhoneSet &a);
};

inline ostream& operator<<(ostream& s, PhoneSet &p)
{
    s << p.phone_set_name(); return s;
}

const EST_String &map_phone(const EST_String &fromphonename,
			    const EST_String &fromsetname,
			    const EST_String &tosetname);
const EST_String &ph_feat(const EST_String &ph,const EST_String &feat);
int ph_is_silence(const EST_String &ph);
int ph_is_vowel(const EST_String &ph);
int ph_is_consonant(const EST_String &ph);
int ph_is_liquid(const EST_String &ph);
int ph_is_approximant(const EST_String &ph);
int ph_is_stop(const EST_String &ph);
int ph_is_nasal(const EST_String &ph);
int ph_is_fricative(const EST_String &ph);
int ph_is_sonorant(const EST_String &ph);
int ph_is_obstruent(const EST_String &ph);
int ph_is_voiced(const EST_String &ph);
int ph_is_sonorant(const EST_String &ph);
int ph_is_syllabic(const EST_String &ph);
int ph_sonority(const EST_String &ph);
EST_String ph_silence(void);

PhoneSet *phoneset_name_to_set(const EST_String &name);

#endif



