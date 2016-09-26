
 /************************************************************************/
 /*                                                                      */
 /*                Centre for Speech Technology Research                 */
 /*                     University of Edinburgh, UK                      */
 /*                        Copyright (c) 1997                            */
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
 /*                 Author: Richard Caley (rjc@cstr.ed.ac.uk)            */
 /*                   Date: February 1997                                */
 /* -------------------------------------------------------------------- */
 /*                                                                      */
 /* Use counted memory chunks and smart pointers to them.                */
 /*                                                                      */
 /************************************************************************/

#if ! defined(__EST_CHUNK_H__)
#define __EST_CHUNK_H__

#define HAVE_WALLOC_H (1)

#include <iostream>
using namespace std;
#include <climits>
#include <sys/types.h>

// Warn when getting a writable version of a shared chunk --
// useful for minimising copies.

/* #define __INCLUDE_CHUNK_WARNINGS__ (1) */

#if defined(__INCULDE_CHUNK_WARNINGS__)
#    define CHUNK_WARN(WHAT) do { cerr << "chunk: " <<WHAT << "\n";} while (0)
#else
#    define CHUNK_WARN(WHAT) // empty
#endif

#define __CHUNK_INLINE_AGGRESSIVELY__ (1)

#if defined(__CHUNK_INLINE_AGGRESSIVELY__)
#    define CII(BODY) BODY
#else
#    define CII(BODY) /* empty */
#endif

#define __CHUNK_USE_WALLOC__ (1)

#if __CHUNK_USE_WALLOC__

#if HAVE_WALLOC_H

# include "EST_walloc.h"

#else

#    define walloc(T,N) ((T *)malloc(sizeof(T)*(N)))
#    define wfree(P) free(P)
#    define wrealloc(P,T,N) ((T *)realloc((P),sizeof(T)*(N)))

#endif

#endif

 /************************************************************************/
 /*                                                                      */
 /* EST_Chunk is a use-counted chunk of memory. You shouldn't be able    */
 /* to do anything to it except create it and manipulate it via          */
 /* EST_ChunkPtr. The private operator::new takes a placement argument   */
 /* which is actually the number of bytes of memory in the body of the   */
 /* chunk.                                                               */
 /*                                                                      */
 /* If the use counter overflows, it sticks. Anything with more than     */
 /* SHRT_MAX references to it is probably permanent.                     */
 /*                                                                      */
 /************************************************************************/

class EST_ChunkPtr;

class EST_Chunk  {
  public:
    typedef  unsigned short use_counter;
#       define MAX_CHUNK_COUNT (USHRT_MAX)
    typedef  int EST_chunk_size;
#       define MAX_CHUNK_SIZE  (INT_MAX)

  private:
    use_counter count;
    EST_chunk_size size;
    int malloc_flag; // set if this was got from malloc (rather than new)
    char  memory[1];

    EST_Chunk(void);
    ~EST_Chunk();

    EST_Chunk *operator & ();
    void *operator new (size_t size, int bytes);
    void operator delete (void *it);
    
    void operator ++ ()
        CII({if (count < MAX_CHUNK_COUNT) ++count; });

    void operator -- ()
        CII({if (count < MAX_CHUNK_COUNT) if (--count == 0) delete this;});

  public:
    friend class EST_ChunkPtr;

    friend EST_ChunkPtr chunk_allocate(int bytes);
    friend EST_ChunkPtr chunk_allocate(int bytes, const char *initial, int initial_len);
    friend EST_ChunkPtr chunk_allocate(int bytes, const EST_ChunkPtr &initial, int initial_start, int initial_len);

    friend void cp_make_updatable(EST_ChunkPtr &shared, EST_chunk_size inuse);
    friend void cp_make_updatable(EST_ChunkPtr &shared);

    friend void grow_chunk(EST_ChunkPtr &shared, EST_chunk_size inuse, EST_chunk_size newsize);
    friend void grow_chunk(EST_ChunkPtr &shared, EST_chunk_size newsize);

    friend ostream &operator << (ostream &s, const EST_Chunk &chp);
    friend void tester(void);
};

 /************************************************************************/
 /*                                                                      */
 /* Pointers to chunks. Initialising them and assigning them around      */
 /* keeps track of use counts. We allow them to be cast to char * as a   */
 /* way of letting people work on them with standard functions,          */
 /* however it is bad voodoo to hold on to such a cast chunk for more    */
 /* than a trivial amount of time.                                       */
 /*                                                                      */
 /************************************************************************/

class EST_ChunkPtr {
  private:
    EST_Chunk *ptr;

    EST_ChunkPtr(EST_Chunk *chp) CII({
      if ((ptr=chp))
	++ *ptr;
    });
  public:
    EST_ChunkPtr(void) { ptr = (EST_Chunk *)NULL; };
    
    EST_ChunkPtr(const EST_ChunkPtr &cp) CII({
      ptr=cp.ptr;
      if (ptr)
	++ *ptr;
    });

    ~EST_ChunkPtr(void) CII({ if (ptr) -- *ptr; });

    int size(void) const { return ptr?ptr->size:0; };
    int shareing(void) const { return ptr?(ptr->count >1):0; };
    int count(void) const { return ptr?(ptr->count):-1; };

    EST_ChunkPtr &operator = (EST_ChunkPtr cp) CII({
      // doing it in this order means self assignment is safe.
      if (cp.ptr)
	++ *(cp.ptr);
      if (ptr)
	-- *ptr;
      ptr=cp.ptr;
      return *this;
    });

    // If they manage to get hold of one...
    // Actually usually used to assign NULL and so (possibly) deallocate
    // the chunk currently pointed to.
    EST_ChunkPtr &operator = (EST_Chunk *chp) CII({
      // doing it in this order means self assignment is safe.
      if (chp)
	++ *chp;
      if (ptr)
	-- *ptr;
      ptr=chp;
      return *this;
    });
 
    // Casting to a non-const pointer causes a
    // warning to stderr if the chunk is shared.
    operator char*() CII({
      if (ptr && ptr->count > 1) 
	{ 
	  CHUNK_WARN("getting writable version of shared chunk\n");
	  cp_make_updatable(*this);
	}
      return ptr?&(ptr->memory[0]):(char *)NULL;
    });
    operator const char*() const CII({
      return ptr?&(ptr->memory[0]):(const char *)NULL;
    });
    operator const char*() CII({
      return ptr?&(ptr->memory[0]):(const char *)NULL;
    });


    const char operator [] (int i) const { return ptr->memory[i]; };
    char &operator () (int i) CII({ 
      if (ptr->count>1) 
	{
	  CHUNK_WARN("getting writable version of shared chunk\n");
	  cp_make_updatable(*this); 
	}
      return ptr->memory[i]; 
    });

    // Creating a new one
    friend EST_ChunkPtr chunk_allocate(int size);
    friend EST_ChunkPtr chunk_allocate(int bytes, const char *initial, int initial_len);
    friend EST_ChunkPtr chunk_allocate(int bytes, const EST_ChunkPtr &initial, int initial_start, int initial_len);

    // Make sure the memory isn`t shared.
    friend void cp_make_updatable(EST_ChunkPtr &shared, EST_Chunk::EST_chunk_size inuse);
    friend void cp_make_updatable(EST_ChunkPtr &shared);

    // Make sure there is enough room (also makes updatable)
    friend void grow_chunk(EST_ChunkPtr &shared, EST_Chunk::EST_chunk_size inuse, EST_Chunk::EST_chunk_size newsize);
    friend void grow_chunk(EST_ChunkPtr &shared, EST_Chunk::EST_chunk_size newsize);

    // we print it by just printing the chunk
    friend ostream &operator << (ostream &s, const EST_ChunkPtr &cp) { return (s<< *cp.ptr); };

    friend void tester(void);
};

#endif
