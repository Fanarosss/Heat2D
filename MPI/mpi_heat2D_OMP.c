#include "mpi.h"
#include "omp.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#define BEGIN       1                  /* message tag */
#define NORTHTAG    2                  /* message tag */
#define SOUTHTAG    3                  /* message tag */
#define WESTTAG     4                  /* message tag */
#define EASTTAG     5                  /* message tag */
#define DONE        4                  /* message tag */
#define MASTER      0                  /* taskid of first process */
#define THREAD_NUM  2

struct Parms {
    float cx;
    float cy;
} parms = {0.1, 0.1};

inline void update(int startx, int endx, int starty, int endy, int ny, float *u1, float *u2);
inline void inidat(int nx, int ny, int offx, int offy, float *u, int NXPROB, int NYPROB);
inline void prtdat(int nx, int ny, float *u1, char *fnam);

int main(int argc, char *argv[]) {
    int taskid, /* this task's unique id */
            numworkers, /* number of worker processes */
            numtasks, /* number of tasks */
            averow, rows, offset, extra, /* for sending rows of data */
            dest, source, /* to - from for message send-receive */
            msgtype, /* for message types */
            rc, start, end, /* misc */
            i, j, ix, iy, iz, it, l,k,
            blocksize,blocksize_x,blocksize_y,
            rowsperblock,
            columnsperblock,
            extrarows,
            extracolumns,
            offsetx, offsety,
            noffsetx, noffsety,
            blockx, blocky,
            currentrows,
            currentcolumns,
            north, south, west, east, /* loop variables */
            startx, endx, starty, endy,
            count_req, halorows, halocolumns,
            temp_rows, temp_cols, worker,
            NXPROB, NYPROB, STEPS;
    double clock_start, clock_end, elapsed_time, max_time;
    MPI_Status status;
    MPI_Request send_requests[2][4];
    MPI_Request recv_requests[2][4];

#ifdef CHECK_CONVERGENCE
    int already_converged = 0, Convergence_iter_num = 30, changes_spotted = 0, overall_check;
#endif

    /* Find out my taskid and how many tasks are running */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    numworkers = numtasks; /* all tasks are going for work */

    blocksize = sqrt(numtasks);
    if (blocksize * blocksize != numtasks) {
        // find factor pairs of numtasks and keep the last
        for (k = 1; k*k <= numtasks; k++) {
            if (numtasks % k == 0){
               blocksize_x = k;
               blocksize_y = numtasks/k;
            }
         }
    }else{
      blocksize_x = blocksize;
      blocksize_y = blocksize;
   }


    /**************** Data Distribution ***************/
    NXPROB = (argv[1] != NULL) ? atoi(argv[1]):-1;
    NYPROB = (argv[2] != NULL) ? atoi(argv[2]):-1;
    STEPS = (argv[3] != NULL) ? atoi(argv[3]):-1;

    if (taskid == 0) {
       if (NXPROB == -1 || NYPROB == -1 || STEPS == -1) {
         printf("Error: Wrong format input!\nCorrect format is NX NY STEPS\nExiting...\n");
         MPI_Abort(MPI_COMM_WORLD, rc);
         exit(1);
      }
      printf("Starting mpi_heat2D with %d worker tasks.\n", numworkers);
      printf("Grid size: X= %d  Y= %d  Time steps= %d\n", NXPROB, NYPROB, STEPS);
    }
    /* calculating general rows and columns per block */
    rowsperblock = NXPROB / blocksize_x;
    extrarows = (NXPROB % blocksize_x);
    columnsperblock = NYPROB / blocksize_y;
    extracolumns = (NYPROB % blocksize_y);

    /* calculating the extra rows and columns */
    blockx = taskid / blocksize_x;
    blocky = taskid % blocksize_y;
    currentrows = (blockx < extrarows) ? rowsperblock + 1 : rowsperblock;
    currentcolumns = (blocky < extracolumns) ? columnsperblock + 1 : columnsperblock;
    /* each worker has max 4 neighbors with whom he has to exchange data with */
    /* neighbor from north */
    north = (taskid - blocksize_y >= 0) ? taskid - blocksize_y : -1;
    /* neighbor from south */
    south = (taskid + blocksize_y < numtasks) ? taskid + blocksize_y : -1;
    /* neighbor from west */
    if (taskid == 0) {
        west = -1;
    } else {
        west = ((taskid - 1) / blocksize_x == blockx) ? taskid - 1 : -1;
    }
    /* neighbor from east */
    if (taskid + 1 == numworkers) {
        east = -1;
    } else {
        east = ((taskid + 1) / blocksize_x == blockx) ? taskid + 1 : -1;
    }
    /* to store the offset of my array on the overall grid */
    offsetx = 0;
    offsety = 0;

    for (l = 1; l < (taskid + 1); l++) {
        blockx = (l - 1) / blocksize_x;
        blocky = (l - 1) % blocksize_y;
        temp_rows = (blockx < extrarows) ? rowsperblock + 1 : rowsperblock;
        temp_cols = (blocky < extracolumns) ? columnsperblock + 1 : columnsperblock;
        if (l / blocksize_x == blockx) {
            offsety += temp_cols;
        } else {
            offsety = 0;
            offsetx += temp_rows;
        }
    }
    /**************** End Of Data Distribution ***************/


    /************** Initializing Structs and Data ************/
    /* Creating cartesian comm */
    int ndims = 2; //dimension of cartesian
    int dims[2] = {blocksize_x, blocksize_y}; //size of every dimension
    int periods[2] = {0, 0}; //is every dim periodical?
    int reorder = 1; //reorder = true
    MPI_Comm cartesian_comm;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cartesian_comm);

    /* processor name check */
    int len;
    char pname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(pname, &len);

    int base_task_offset = offsety * NYPROB + offsetx;

    /* initializing my grid - dynamically allocate memory for 3D array */

    halorows = currentrows + 2;
    halocolumns = currentcolumns + 2;
    float u[2][halorows][halocolumns];

    /* initializing array */
    for (iz = 0; iz < 2; iz++)
        for (ix = 0; ix < halorows; ix++)
            for (iy = 0; iy < halocolumns; iy++)
                u[iz][ix][iy] = 0.0;
    /* initializing data types for halo point exchange */
    MPI_Datatype column_type;
    MPI_Type_vector(currentrows, 1, halocolumns, MPI_FLOAT, &column_type);
    MPI_Type_commit(&column_type);
    /* fill my grid with data */
    inidat(currentrows, currentcolumns, offsetx, offsety, u, NXPROB, NYPROB);
    /*************** End Of Initialization ****************/

    /* Task synchronization */
    MPI_Barrier(MPI_COMM_WORLD);
    /* Clock Start */
    clock_start = MPI_Wtime();

    /*** Setting neighbors outside the loop ***/
    iz = 0;
    count_req = 4;
    if (north == -1) {
        north = MPI_PROC_NULL;
    }
    if (south == -1) {
        south = MPI_PROC_NULL;
    }
    if (west == -1) {
        west = MPI_PROC_NULL;
    }
    if (east == -1) {
        east = MPI_PROC_NULL;
    }

    MPI_Send_init(&u[0][1][1], currentcolumns, MPI_FLOAT, north, SOUTHTAG, cartesian_comm, &send_requests[0][0]);
    MPI_Send_init(&u[0][halorows - 2][1], currentcolumns, MPI_FLOAT, south, NORTHTAG, cartesian_comm, &send_requests[0][1]);
    MPI_Send_init(&u[0][1][1], 1, column_type, west, EASTTAG, cartesian_comm, &send_requests[0][2]);
    MPI_Send_init(&u[0][1][halocolumns - 2], 1, column_type, east, WESTTAG, cartesian_comm, &send_requests[0][3]);

    MPI_Recv_init(&u[0][0][1], currentcolumns, MPI_FLOAT, north, NORTHTAG, cartesian_comm, &recv_requests[0][0]);
    MPI_Recv_init(&u[0][halorows - 1][1], currentcolumns, MPI_FLOAT, south, SOUTHTAG, cartesian_comm, &recv_requests[0][1]);
    MPI_Recv_init(&u[0][1][0], 1, column_type, west, WESTTAG, cartesian_comm, &recv_requests[0][2]);
    MPI_Recv_init(&u[0][1][halocolumns - 1], 1, column_type, east, EASTTAG, cartesian_comm, &recv_requests[0][3]);

    MPI_Send_init(&u[1][1][1], currentcolumns, MPI_FLOAT, north, SOUTHTAG, cartesian_comm, &send_requests[1][0]);
    MPI_Send_init(&u[1][halorows - 2][1], currentcolumns, MPI_FLOAT, south, NORTHTAG, cartesian_comm, &send_requests[1][1]);
    MPI_Send_init(&u[1][1][1], 1, column_type, west, EASTTAG, cartesian_comm, &send_requests[1][2]);
    MPI_Send_init(&u[1][1][halocolumns - 2], 1, column_type, east, WESTTAG, cartesian_comm, &send_requests[1][3]);

    MPI_Recv_init(&u[1][0][1], currentcolumns, MPI_FLOAT, north, NORTHTAG, cartesian_comm, &recv_requests[1][0]);
    MPI_Recv_init(&u[1][halorows - 1][1], currentcolumns, MPI_FLOAT, south, SOUTHTAG, cartesian_comm, &recv_requests[1][1]);
    MPI_Recv_init(&u[1][1][0], 1, column_type, west, WESTTAG, cartesian_comm, &recv_requests[1][2]);
    MPI_Recv_init(&u[1][1][halocolumns - 1], 1, column_type, east, EASTTAG, cartesian_comm, &recv_requests[1][3]);

    /************* Begin STEPS iterations. *************/
    for (it = 1; it <= STEPS; it++) {
        /* Start the communication */
	MPI_Startall(count_req, recv_requests[iz]);
	MPI_Startall(count_req, send_requests[iz]);

        /* Call update for the inside values, that do not need anything else to be updated */
        update(2, currentrows - 1, 2, currentcolumns - 1, halocolumns, &u[iz][0][0], &u[1 - iz][0][0]);

        /* Wait to get the halo points */
	MPI_Waitall(count_req, recv_requests[iz], MPI_STATUS_IGNORE);

        /* update north outsiders */
        update(1, 1, 1, currentcolumns, halocolumns, &u[iz][0][0], &u[1 - iz][0][0]);

        /* update west outsiders */
        update(1, currentrows, 1, 1, halocolumns, &u[iz][0][0], &u[1 - iz][0][0]);

        /* update south outsiders */
        update(currentrows, currentrows, 1, currentcolumns, halocolumns, &u[iz][0][0], &u[1 - iz][0][0]);

        /* update east outsiders */
        update(1, currentrows, currentcolumns, currentcolumns, halocolumns, &u[iz][0][0], &u[1 - iz][0][0]);

        /* Now finally wait for our send request to pass through */
        MPI_Waitall(count_req, send_requests[iz], MPI_STATUS_IGNORE);


        /* Check for Convergence */
#ifdef CHECK_CONVERGENCE
        float abs0, abs1;
        if (it % Convergence_iter_num == 0 && already_converged == 0) {
            changes_spotted = 0;
            for (i = 1; i <= currentrows; i++) {
                for (j = 1; j <= currentcolumns; j++) {
                    abs0 = abs(u[0][i][j]);
                    abs1 = abs(u[1][i][j]);
                    if(abs0 != 0 && abs1 != 0 && (abs0 + abs1 >= FLT_MIN)){
                        changes_spotted = 1;
                        break;
                    }
                }
                if (changes_spotted == 1) {
                    break;
                }
            }
            MPI_Allreduce(&changes_spotted, &overall_check, 1, MPI_INT, MPI_SUM, cartesian_comm);
            if (overall_check == 0) {
                if (taskid == 0)
                  printf("Grid Convergence at Iteration check: %d\n", it);
                already_converged = 1;
            }
        }
#endif
        /* End of Check */

        /* before = after */
        iz = 1 - iz;
    }
    /************* End Of Iterations. *************/

    /* Stop clock and get the elapsed time */
    clock_end = MPI_Wtime();
    elapsed_time = clock_end - clock_start;
    /* To find the max time between all tasks */
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, cartesian_comm);
    /* end of clock */


    /* parallel write */
#ifdef WRITE_FILE
    MPI_File fh;

    MPI_File_open(MPI_COMM_WORLD, "parallel_final.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);


    for (i = 1; i <= currentrows; i++) {
        int writing_offset = sizeof (float)*((offsetx + i - 1) * NYPROB + offsety);
        MPI_File_seek(fh, writing_offset, MPI_SEEK_SET);
        MPI_File_write(fh, &u[iz][i][1], currentcolumns, MPI_FLOAT, &status);
    }

    /* end of writing */
    MPI_File_close(&fh);
#endif


    /* free memory */
    MPI_Type_free(&column_type);

    MPI_Comm_free(&cartesian_comm);
    /* end of free */


    if (taskid == 0) {

#ifdef VERIFY_FILE

        FILE* fp = fopen("parallel_final.dat", "rb");

        if (!fp) {
            printf("Master: file not found \n");
        } else {
            printf("Master: Verifying output file ... \n");

            for (i = 1; i <= NXPROB; i++) {
                for (j = 1; j <= NYPROB; j++) {
                    float temp = -1;
                    if (fread(&temp, sizeof (float), 1, fp) == 0) {
                        printf("Error \n");
                        return 1;
                    }
                    printf("%6.1f ", temp);
                }
                printf("\n");
            }

            fclose(fp);
        }
#endif

        int qq = 0, ww = 0, ff = 0;
#ifdef OPENMP_ON
        qq = 1;
#endif
#ifdef CHECK_CONVERGENCE
        ww = 1;
#endif
#ifdef WRITE_FILE
       ff = 1;
#endif
       printf("[Heat2Dn in Parallel Ended, OMP:%d, CON:%d, I|O:%d]: \n\tSize_X: %d Size_Y: %d \n\tSteps: %d \n\tTime: %.10f seconds\n", qq, ww, ff, NXPROB, NYPROB, STEPS, max_time);

    }


    MPI_Finalize();



    return 0;
}

/**************************************************************************
 *  subroutine update
 ****************************************************************************/
void update(int startx, int endx, int starty, int endy, int ny, float *u1, float *u2) {
    int ix, iy;
    for (ix = startx; ix <= endx; ix++)
        for (iy = starty; iy <= endy; iy++)
            if (*(u1 + ix * ny + iy) != 0.0) {
                *(u2 + ix * ny + iy) = *(u1 + ix * ny + iy) +
                        parms.cx * (*(u1 + (ix + 1) * ny + iy) +
                        *(u1 + (ix - 1) * ny + iy) -
                        2.0 * *(u1 + ix * ny + iy)) +
                        parms.cy * (*(u1 + ix * ny + iy + 1) +
                        *(u1 + ix * ny + iy - 1) -
                        2.0 * *(u1 + ix * ny + iy));
            }
}

/*****************************************************************************
 *  subroutine inidat
 *****************************************************************************/

void inidat(int nx, int ny, int offx, int offy, float *u, int NXPROB, int NYPROB) {
    int ix, iy, dx, dy;
    for (ix = 0; ix <= nx + 1; ix++) {
        for (iy = 0; iy <= ny + 1; iy++) {
            dx = ix + offx - 1;
            dy = iy + offy - 1;
            if (ix == 0 || iy == 0 || ix == nx + 1 || iy == ny + 1) {
                *(u + ix * (ny + 2) + iy) = 0.0;
                continue;
            }
            if (dx == 0 || dx == NXPROB || dy == 0 || dy == NYPROB) {
                *(u + ix * (ny + 2) + iy) = 0.0;
            } else {
                *(u + ix * (ny + 2) + iy) = (float) (dx * (NXPROB - dx - 1) * dy * (NYPROB - dy - 1));
            }
        }
    }
}

/**************************************************************************
 * subroutine prtdat
 **************************************************************************/
void prtdat(int nx, int ny, float *u1, char *fnam) {
    int ix, iy;
    FILE *fp;

    if (fnam != NULL) {
        fp = fopen(fnam, "w");
    } else {
        fp = stdout;
    }
    for (ix = 1; ix <= nx; ix++) {
        for (iy = 1; iy <= ny; iy++) {
            fprintf(fp, "%6.1f", *(u1 + ix * (ny + 2) + iy));
            if (iy != ny)
                fprintf(fp, " ");
            else
                fprintf(fp, "\n");
        }
    }

    if (fp != stdout) {
        fclose(fp);
    }
}
