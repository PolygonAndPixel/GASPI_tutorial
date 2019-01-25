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

#include "aux/success_or_die.h"

#include <mpi.h>
#include <GASPI.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{ 
  MPI_Init (&argc, &argv);
  SUCCESS_OR_DIE( gaspi_proc_init(GASPI_BLOCK) );
  gaspi_rank_t rank;
  gaspi_rank_t num;
  
  SUCCESS_OR_DIE( gaspi_proc_rank(&rank) );
  SUCCESS_OR_DIE( gaspi_proc_num(&num) );
 
  printf("Hello world from rank %d of %d\n",rank, num);
 
  SUCCESS_OR_DIE( gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK) );
  SUCCESS_OR_DIE( gaspi_proc_term(GASPI_BLOCK) );
  MPI_Finalize();
  return EXIT_SUCCESS;
}
