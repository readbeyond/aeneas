/*************************************************************************/
/*                                                                       */
/*                Centre for Speech Technology Research                  */
/*                     University of Edinburgh, UK                       */
/*                         Copyright (c) 1995,1996                       */
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
/*                    Date   :  July 1995                                */
/*-----------------------------------------------------------------------*/
/*                   Event class header file                             */
/*                                                                       */
/*=======================================================================*/

// Warning: These event classes can be used as items in the EST_Stream
// int he normal way. However, the EventSI class has internal pointers to
// adjacent events which it uses to work out various parameters. Thus,
// when making a EST_Stream of events, it is important to link the events
// pointers as well.

#ifndef __Event_H__
#define __Event_H__

#include <EST_String.h>

#ifndef FALSE
#       define  FALSE   (0)
#endif
#ifndef TRUE
#       define  TRUE    (1)
#endif


//class RFCelement{
//    float amp;
//    float dur;
//    float start_amp;
//    float start_pos;
//    EST_String type;
//};

class EventBase{
 private:
 public:
    EST_String type;
    int save(EST_String filename, EST_String type = "");
    
};

class EventRFC: public EventBase{
public:
    void init() {rise_amp = 0.0; rise_dur = 0.0; 
	    start_amp =0.0; fall_amp = 0.0; fall_dur = 0.0; 
	    start_pos = 0.0; peak_pos = 0.0; type = ""; }

    float rise_amp;
    float rise_dur;
    float fall_amp;
    float fall_dur;
    float peak_pos;
    float start_amp;
    float start_pos;
    friend ostream& operator << (ostream& s, EventRFC &e) 
    {
	s << e.type << " " << e.rise_amp << " " << e.rise_dur
	    << " " << e.fall_amp << " " << e.fall_dur
	    << " " << e.start_amp << " " << e.start_pos
            << endl;
	    return s;
	}
};

class EventSI: public EventBase {
private:
    float s_f0;
    float s_pos;
    float p_f0;
    float p_pos;
    float e_pos;
    float e_f0;

public:
    void init();

    float amp();
    float dur();

    float rise_amp();
    float rise_dur();
    float fall_amp();
    float fall_dur();

    float start_f0();
    float start_pos();
    float peak_f0();
    float peak_pos();
    float end_f0();
    float end_pos();

    void set_start_f0(float a);
    void set_start_pos(float a);
    void set_peak_f0(float a);
    void set_peak_pos(float a);
    void set_end_f0(float a);
    void set_end_pos(float a);

    EventSI *sn;
    EventSI *sp;

    friend ostream& operator << (ostream& s, EventSI &e) 
    {
	s << e.type << " ra:" << e.rise_amp() << " rd:" << e.rise_dur()
	    << "fa: " << e.fall_amp() << " fd:" << e.fall_dur()
	    << " sf0:" << e.start_f0() << " spos:" << e.start_pos()
	    << " pf0:" << e.peak_f0() << " ppos:" << e.peak_pos()
	    << " ef0:" << e.end_f0() << " epos:" << e.end_pos()
            << endl;
	    return s;
	}
};

class EventTilt: public EventBase{
private:
    float samp;
    float sdur;
    float stilt;
    float spos;
    float s_f0;
    float s_pos;

public:
    void init();

    float amp();
    float dur();
    float tilt();
    float pos();

    void set_amp(float a);
    void set_dur(float a);
    void set_tilt(float a);
    void set_pos(float a);

    float start_f0();
    float start_pos();
    void set_start_f0(float a);
    void set_start_pos(float a);

    friend ostream& operator << (ostream& s, EventTilt &e) 
    {
	s << e.type << " " << e.amp() << " " << e.dur()
	    << " " << e.tilt() << " " << e.pos()
	    << " sf0 " << e.start_f0() << " " << e.start_pos()
            << endl;
	    return s;
	}

};

void gc_eventsi(void *w);
void gc_eventtilt(void *w);
void gc_eventrfc(void *w);

#endif // __Event_H__
