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

#ifndef __EST_TRACKMAP_H__
#define __EST_TRACKMAP_H__

#include <climits>
using namespace std;

#include "EST_TNamedEnum.h"
#include "EST_ChannelType.h"
#include "EST_Handleable.h"
#include "EST_THandle.h"

/** Track maps provide a mapping from symbolic track names to the
  * actual position of the information within a track frame. The
  * symbolic names are defined by the EST_ChannelType enumerated type.
  * 
  * Track maps can be declared statically by code which always uses
  * tracks of a given style, or they can be built at run time as
  * is done by lpc_analysis to record whichinformation the
  * user has requested. Finally they can be constructed by the Track
  * itself from the names of the channels, for instance when a track has
  * just been read in from a file.
  *
  * @see EST_Track
  * @see EST_ChannelType
  * @see EST_TrackMap:example
  * @author Richard Caley <rjc@cstr.ed.ac.uk>
  * @version $Id: EST_TrackMap.h,v 1.4 2004/09/29 08:24:17 robert Exp $
  */
class EST_TrackMap : public EST_Handleable
{

public:
  /**@name ChannelMapping
    * An auxiliary type used just to define static EST_TrackMaps.
    * Defining one of these and then converting it to an EST_TrackMap
    * is, unfortunately, the only way C++ allows us to define
    * a constant EST_TrackMap.
    */
  //@{
  /// structure for the table.
  struct ChannelMappingElement {
    EST_ChannelType type;
    unsigned short channel;
  };
  /// Table of type to position pairs.
//  typedef struct ChannelMappingElement ChannelMapping[];
  //@}

  typedef EST_THandle<EST_TrackMap,EST_TrackMap> P;

public:

  /// Returned if we ask for a channel not in the map.
#   define NO_SUCH_CHANNEL (-1)

private:

  /// The map itself.
  short p_map[num_channel_types];

  /// Parent is looked at if this map doesn't define the position.
  EST_TrackMap::P p_parent;
  /// Subtracted from the values in the parent.
  int p_offset;

  /// No copy constructor. Don't copy these things.
  EST_TrackMap(EST_TrackMap &from);

protected:
  /// Pass to creation function to turn on refcounting.
#define EST_TM_REFCOUNTED (1)

  /// Creation function used by friends to create refcounted maps.
  EST_TrackMap(int refcount);	

  /// Creation function used by friends to create sub-track maps.
  EST_TrackMap(const EST_TrackMap *parent, int offset, int refcount);

  /// copy an exiting map.
  void copy(EST_TrackMap &from);
  /// Initialise the map.
  void init(void);

  short get_parent(EST_ChannelType type) const ;

public:
  /// Default constructor.
  EST_TrackMap(void);
  /// Copy the mapping.
  EST_TrackMap(EST_TrackMap &from, int refcount);
  /// Create from static table.
  EST_TrackMap(struct ChannelMappingElement map[]);

  
  ~EST_TrackMap();

  /// Empty the map.
  void clear(void);
  /// Record the position of a channel.
  void set(EST_ChannelType type, short pos) 
	{ p_map[(int)type] = pos; }
  
  /// Get the position of a channel.
  short get(EST_ChannelType type) const 
	{ short c = p_map[(int)type]; 
	return c!=NO_SUCH_CHANNEL?c:get_parent(type); }
  /// Get the position of a channel.
  short operator() (EST_ChannelType type) const 
	{ return get(type); }

  /// Does the mapping contain a position for this channel?
  bool has_channel(EST_ChannelType type) const 
	{ return p_map[(int)type] != NO_SUCH_CHANNEL
	    || ( p_parent!=0 && p_parent->has_channel(type) ); }

  /// Returns the index of the last known channel.
  short last_channel(void) const;

  /// Returns the type of the channel at the given position.
  EST_ChannelType channel_type(unsigned short channel) const;

  EST_TrackMap * object_ptr() { return this; }
  const EST_TrackMap * object_ptr() const { return this; }

  friend class EST_Track;
  friend ostream& operator << (ostream &st, const EST_TrackMap &m);
};

/** Channel name maps map textual names for track channels to symbolic
  * names, they are just a special case of named enums.
  */
typedef EST_TNamedEnum<EST_ChannelType> EST_ChannelNameMap;

/** Table type used to create EST_ChannelNameMaps.
  */
typedef EST_TValuedEnumDefinition<EST_ChannelType, const char *, NO_INFO> 
	EST_ChannelNameTable[];

/// Definition of standard names we use for channels.
extern EST_ChannelNameMap EST_default_channel_names;
/// Definition of the names ESPS programs use for channels.
extern EST_ChannelNameMap esps_channel_names;

extern ostream& operator << (ostream &st, const EST_TrackMap &m);
#endif
