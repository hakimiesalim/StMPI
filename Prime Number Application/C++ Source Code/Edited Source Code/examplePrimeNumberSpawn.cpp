/******************************************************************************
* FILE: mpi_prime.c
* DESCRIPTION:
*   Generates prime numbers.  All tasks distribute the work evenly, taking
*   every nth number, where n is the stride computed as:  (rank *2) + 1
*   so that even numbers are automatically skipped.  The method of using
*   stride is preferred over contiguous blocks of numbers, since numbers
*   in the higher range require more work to compute and may result in
*   load imbalance.  This program demonstrates embarrassing parallelism.
*   Collective communications calls are used to reduce the only two data
*   elements requiring communications: the number of primes found and
*   the largest prime.
* AUTHOR: Blaise Barney 11/25/95 - adapted from version contributed by
*   Richard Ng &  Wong Sze Cheong during MHPCC Singapore Workshop (8/22/95).
* LAST REVISED: 04/13/05
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define FIRST     0           /* Rank of first task */

int isprime(int n) {
    int i, squareroot;
    if (n > 10) {
        squareroot = (int)sqrt(n);
        for (i = 3; i <= squareroot; i = i + 2)
            if ((n % i) == 0)
                return 0;
        return 1;
    }
    /* Assume first four primes are counted elsewhere. Forget everything else */
    else
        return 0;
}


int main(int argc, char* argv[])
{
    int   ntasks,               /* total number of tasks in partitiion */
          rank,                 /* task identifier */
          n,                    /* loop variable */
          pc,                   /* prime counter */
          pcsum,                /* number of primes found by all tasks */
          foundone,             /* most recent prime found */
          maxprime,             /* largest prime found */
          mystart,              /* where to start calculating */
          stride;               /* calculate every nth number */
    long  LIMIT;                /*limit of the prime numbers*/

    MPI_Comm parentcomm;        //parent communicator 

    MPI_Init(&argc, &argv);     //initialize MPI environment    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //rank the process
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);  //size of MPI world

    /* Get parent communicator*/
    MPI_Comm_get_parent(&parentcomm);

    /* Receive LIMIT */
    MPI_Bcast(&LIMIT, 1, MPI_LONG, 0, parentcomm);

    mystart = (rank * 2) + 1;       /* Find my starting point - must be odd number */
    stride = ntasks * 2;          /* Determine stride, skipping even numbers */
    pc = 0;                       /* Initialize prime counter */
    foundone = 0;               /* Initialize */

    /******************** task with rank 0 does this part ********************/
    if (rank == FIRST) {
        printf("Using %d tasks to scan %d numbers\n", ntasks, LIMIT);
        pc = 4;                  /* Assume first four primes are counted here */
        for (n = mystart; n <= LIMIT; n = n + stride) {
            if (isprime(n)) {
                pc++;
                foundone = n;
                /***** Optional: print each prime as it is found
                printf("%d\n",foundone);
                *****/
            }
        }
        MPI_Reduce(&pc, &pcsum, 1, MPI_INT, MPI_SUM, FIRST, MPI_COMM_WORLD);
        MPI_Reduce(&foundone, &maxprime, 1, MPI_INT, MPI_MAX, FIRST, MPI_COMM_WORLD);
   
        MPI_Send(&maxprime, 1, MPI_INT, 0, 0, parentcomm);  //send the result to the smalltalk master
        MPI_Send(&pcsum, 1, MPI_INT, 0, 0, parentcomm);     //send the result to the smalltalk master
    }


    /******************** all other tasks do this part ***********************/
    if (rank > FIRST) {
        for (n = mystart; n <= LIMIT; n = n + stride) {
            if (isprime(n)) {
                pc++;
                foundone = n;
                /***** Optional: print each prime as it is found
                printf("%d\n",foundone);
                *****/
            }
        }
        MPI_Reduce(&pc, &pcsum, 1, MPI_INT, MPI_SUM, FIRST, MPI_COMM_WORLD);
        MPI_Reduce(&foundone, &maxprime, 1, MPI_INT, MPI_MAX, FIRST, MPI_COMM_WORLD);
    }

    MPI_Comm_disconnect(&parentcomm);  //disconnect from Smalltalk master
    MPI_Finalize();
}