/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                         Copyright (c) 1994,1995,1996                  */
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
/*                    Author :  Paul Taylor                              */
/*                    Date   :  May 1994                                 */
/*                    Release:  0.9                                      */
/*-----------------------------------------------------------------------*/
/*              Auxiliary Label Routines header file                     */
/*                                                                       */
/*=======================================================================*/

#ifndef __EST_RELATION_AUX_H__
#define __EST_RELATION_AUX_H__

#include "EST_String.h"
#include "EST_Track.h"
#include "ling_class/EST_Utterance.h"
#include "EST_Option.h"
#include "EST_THash.h"

typedef EST_TList<EST_Relation> EST_RelationList;
typedef EST_TStringHash<EST_Relation*> EST_hashedRelationList;

EST_String options_relation_filetypes(void);


void convert_to_broad(EST_Relation &seg, EST_StrList &pos_list, 
		      EST_String broad_name ="", int polarity = 1);
void convert_to_broad_class(EST_Relation &seg, const EST_String &class_type, 
			   EST_Option &options);

int merge_label(EST_Relation &seg, const EST_String &labtype);

void change_label(EST_Relation &seg, const EST_String &oname, const EST_String &nname);

void merge_all_label(EST_Relation &seg, const EST_String &labtype);

void track_to_label(const EST_Track &tr, EST_Relation &lab, float thresh=0.0);
void track_to_pm(const EST_Track &tr, int sample_rate, EST_Relation &lab);

void label_to_track(const EST_Relation &lab, 
		    const EST_Option &al, 
		    const EST_Option &op,
		    EST_Track &tr);
void label_to_track(const EST_Relation &lab, EST_Track &tr,
		    float shift, float offset=0.0, 
		    float range = 1.0, float req_length = -1.0, 
		    const EST_String &pad="low");

void shift_label(EST_Relation &seg, float shift);
void label_map(EST_Relation &seg, EST_Option &map);
void quantize(EST_Relation &a, float q);
int edit_labels(EST_Relation &a, EST_String sedfile);

void RelationList_select(EST_RelationList &mlf, EST_StrList filenames, bool
			exact_match);
EST_Relation RelationList_extract(EST_RelationList &mlf, 
			      const EST_String &filename, 
			      bool base);
EST_Relation RelationList_combine(EST_RelationList &mlf);
EST_Relation RelationList_combine(EST_RelationList &mlf, EST_Relation &key);

int relation_divide(EST_RelationList &mlf, EST_Relation &lab, 
		  EST_Relation &keylab, EST_String ext);

int relation_divide(EST_RelationList &mlf, EST_Relation &lab, 
		    EST_Relation &keylab,
		    EST_StrList &list, EST_String ext);

EST_Litem *RelationList_ptr_extract(EST_RelationList &mlf, 
				const EST_String &filename, 
				bool base);

void relation_convert(EST_Relation &lab, EST_Option &al, EST_Option &op);

EST_read_status load_RelationList(const EST_String &filename, 
				EST_RelationList &plist);
EST_write_status save_RelationList(const EST_String& filename, 
				 const EST_RelationList &plist);
EST_write_status save_RelationList(const EST_String &filename, 
				 const EST_RelationList &plist, 
				 int time=1, int path = 1);
EST_write_status save_ind_RelationList(const EST_String &filename, 
				     const EST_RelationList &plist, 
				     const EST_String &features, 
				     int path);

EST_write_status save_WordList(const EST_String &filename, 
			       const EST_RelationList &plist, 
			       int n);

EST_write_status save_SentenceList(EST_String filename, 
				   EST_RelationList &plist, int n);



EST_read_status read_RelationList(EST_RelationList &mlf, 
				EST_StrList &files, EST_Option &al);

float start(EST_Item *n);
float duration(EST_Item *n);

/// hashed relation lists for super speed
void build_RelationList_hash_table(EST_RelationList &mlf,
				   EST_hashedRelationList &hash_table, 
				   const bool base);

bool hashed_RelationList_extract(EST_Relation* &rel,
				 const EST_hashedRelationList &hash_table,
				 const EST_String &filename, bool base);


#endif /* __EST_RELATION_AUX_H__ */
