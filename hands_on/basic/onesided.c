/*
 * This file is part of a small series of tutorial,
 * which aims to demonstrate key features of the GASPI
 * standard by means of small but expandable examples.
 * Conceptually the tutorial follows a MPI course
 * developed by EPCC and HLRS.
 *
 * Contact point for the MPI tutorial:
 *                 rabenseifner@hlrs.de
 * Contact point for the GASPI tutorial:
 *                 daniel.gruenewald@itwm.fraunhofer.de
 *                 mirko.rahn@itwm.fraunhofer.de
 *                 christian.simmendinger@t-systems.com
 */

#include "left_right.h"
#include "aux/queue.h"
#include "aux/success_or_die.h"
#include "aux/waitsome.h"

#include <mpi.h>
#include <GASPI.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  static const int VLEN = 4;
  MPI_Init (&argc, &argv);
  SUCCESS_OR_DIE( gaspi_proc_init(GASPI_BLOCK) );
 
  gaspi_rank_t iProc, nProc;
  SUCCESS_OR_DIE( gaspi_proc_rank(&iProc) );
  SUCCESS_OR_DIE( gaspi_proc_num(&nProc) );

  gaspi_segment_id_t const segment_id = 0;
  gaspi_size_t       const segment_size = 2 * VLEN * sizeof (double);

  SUCCESS_OR_DIE
    ( gaspi_segment_create
      ( segment_id, segment_size
      , GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_UNINITIALIZED
      )
    );
    
  gaspi_pointer_t array;
  SUCCESS_OR_DIE( gaspi_segment_ptr (segment_id, &array) );
 
  double * src_array = (double *)(array);
  double * rcv_array = src_array + VLEN;

  for (int j = 0; j < VLEN; ++j)
    {
      src_array[j]= (double)( iProc * VLEN + j );
    }

  gaspi_notification_id_t data_available = 0;
  gaspi_queue_id_t queue_id = 0;
 
  gaspi_offset_t loc_off = 0;
  gaspi_offset_t rem_off = VLEN * sizeof (double);

  /*
    TODO
    ====
    - check queue for overflow before posting a request
    - gaspi_write_notify 1-dim array of size VLEN to right neighbour
    - wait for notification from left neighbour
    - what happens if queue is overflowing: normal / debug library
    
    PLAY AROUND:
    - what happens if you do not use the notification mechanism?
    - what happens if your data transfer is out of bounds?
  */

  for (int j = 0; j < VLEN; ++j)
    {
      printf("rank %d rcv elem %d: %f \n", iProc,j,rcv_array[j] );
    }
  
  SUCCESS_OR_DIE( gaspi_wait ( queue_id, GASPI_BLOCK ) );
 
  SUCCESS_OR_DIE( gaspi_proc_term(GASPI_BLOCK) );
  MPI_Finalize();
  return EXIT_SUCCESS;
}
