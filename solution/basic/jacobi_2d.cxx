#include <iostream>
#include <cstdint>
#include <vector>
#include <limits>
#include <string>
#include <algorithm>

#include <assert.h>
#include <mpi.h>

#include "aux/queue.h"
#include "aux/success_or_die.h"
#include "aux/waitsome.h"
#include "aux/success_or_die.h"

#include <GASPI.h>

#include "aux/bitmap_IO.hpp"

// sponsored by stack overflow: http://stackoverflow.com/questions/440133
std::string random_string(size_t length) {

    auto randchar = []() -> char {
        const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
        return charset[ rand() % (sizeof(charset) - 1)];
    };

    srand(time(0));
    std::string str(length, 0);
    std::generate_n(str.begin(), length, randchar);
    return str;
}

int main (int argc, char * argv[]) {

    // Invoke MPI::Init before gaspi_proc_init !!!
    MPI::Init (argc, argv);

    SUCCESS_OR_DIE( gaspi_proc_init(GASPI_BLOCK) );

    const uint64_t height    =   3*4*5*6+2;
    const uint64_t width     = 2*3*4*5*6+2;
    const uint64_t num_ranks = MPI::COMM_WORLD.Get_size();
    const uint64_t     rank  = MPI::COMM_WORLD.Get_rank();
    const uint64_t radix     = uint64_t(std::sqrt(num_ranks));
    const uint64_t verti     = rank / radix;
    const uint64_t horiz     = rank % radix;
    const bool     is_root   = rank == 0;

    assert(num_ranks == radix*radix);
    assert((height-2) % radix == 0);
    assert((width -2) % radix == 0);
    
    //////////////////////////////////////////////////////////////////////////
    // GASPI SEGMENT 
    //////////////////////////////////////////////////////////////////////////
    gaspi_segment_id_t const segment_id = 0;
    gaspi_segment_id_t const segment_id_root = 1;

    const uint64_t local_height = (height-2)/radix+2;
    const uint64_t local_width  = (width -2)/radix+2;
    
    // Create gaspi segments here for ying and yang
    SUCCESS_OR_DIE( gaspi_segment_create(
        segment_id,
        local_height*local_width*2*sizeof(double), 
        GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED
      )
    );
    // Create pointer to segment
    gaspi_pointer_t array;
    SUCCESS_OR_DIE( gaspi_segment_ptr (segment_id, &array) );
    // Use ying and yang again
    double * ying = (double *) (array);
    double * yang = ying + local_height*local_width;
    
    // make indexing great again!
    auto       at = [&] (const uint64_t& row, const uint64_t& col) {
        return row*width+col;
    };
    auto local_at = [&] (const uint64_t& row, const uint64_t& col) {
        return row*local_width+col;
    };
    double * image = nullptr;
    if(is_root)
    {
        SUCCESS_OR_DIE( gaspi_segment_alloc(
            segment_id_root, 
            height*width*sizeof(double),
            GASPI_MEM_UNINITIALIZED)
        );
        SUCCESS_OR_DIE( gaspi_segment_register(
            segment_id_root,
            rank,
            GASPI_BLOCK)
        );
        
        gaspi_pointer_t image_gaspi_ptr;
        SUCCESS_OR_DIE( gaspi_segment_ptr (segment_id_root, &image_gaspi_ptr) );
        image = (double *) (image_gaspi_ptr);
        
        // draw a checkerboard
        const uint64_t stride = 181;
        for (uint64_t row = 0; row < height; ++row)
            for (uint64_t col = 0; col < width; ++col)
                image[at(row, col)] = (row/stride + col/stride) % 2;
    }
 
    //////////////////////////////////////////////////////////////////////////
    // SCATTER
    //////////////////////////////////////////////////////////////////////////

    // We always write to the same segment
    std::vector<gaspi_segment_id_t> segment_id_vec(local_height, segment_id);
    // We read from segment 1 all the time
    std::vector<gaspi_segment_id_t> segment_id_image_vec(local_height, segment_id_root);
    std::vector<gaspi_offset_t> offset_local(local_height);
    std::vector<gaspi_offset_t> offset_remote(local_height);
    std::vector<gaspi_size_t> size(local_height, local_width*sizeof(double));
    
    gaspi_notification_id_t data_available = 0;
    gaspi_queue_id_t queue_id = 0;
    // Only root distributes data
    if(is_root)
    {
        gaspi_offset_t offset_basis_remote = 0;//(local_width+1)*sizeof(double);
        for(auto &off: offset_remote)
        {
            off = offset_basis_remote;
            offset_basis_remote += local_width*sizeof(double);
        }
        for(gaspi_rank_t next_rank=1; next_rank<num_ranks; ++next_rank)
        {
            // The local offset for the current tile
            gaspi_offset_t offset_basis_local = 
                ( next_rank % radix ) * (local_width-2) 
                + ( next_rank / radix ) * (local_height-2)*width;

            // Always think in bytes!
            offset_basis_local *= sizeof(double);

            // Add the offset for every local_row to send
            for(auto &off: offset_local)
            {
                off = offset_basis_local;
                offset_basis_local += width*sizeof(double); 
            }

            // Send the data
            write_list_notify_and_wait(
                local_height, segment_id_image_vec.data(), offset_local.data(),
                next_rank, segment_id_vec.data(), offset_remote.data(), 
                size.data(), data_available, next_rank+1, queue_id);
        }
        // We also need the data in the segment of root 
        gaspi_offset_t offset_basis_local = 0;
        for(auto &off: offset_local)
        {
            off = offset_basis_local;
            offset_basis_local += width*sizeof(double); 
        }


        
        write_list_notify_and_wait(
            local_height, segment_id_image_vec.data(), offset_local.data(), 
            0, segment_id_vec.data(), offset_remote.data(), 
            size.data(), data_available, 1, queue_id);
    }

    // Every rank waits for its data and starts working 
    wait_or_die(segment_id, data_available, rank+1);

    //////////////////////////////////////////////////////////////////////////
    // FIXPOINT COMPUTATION
    //////////////////////////////////////////////////////////////////////////

    auto update = [&] (const uint64_t& row, const uint64_t& col) {
        return 0.25*(ying[local_at(row+1, col)] + ying[local_at(row-1, col)]
                   + ying[local_at(row, col+1)] + ying[local_at(row, col-1)]);
    };

	// Halo above and below: local_height-2
    // Data per row: 1
    // Stride: local_width (duh!)
    MPI::Datatype col_t = 
    	MPI::DOUBLE.Create_vector (local_height-2, 1, local_width)
                   .Create_resized(0, sizeof(double));
    col_t.Commit();

    uint64_t counter = 0, print_every = 1024;
    double error = std::numeric_limits<double>::infinity();

	uint64_t halo = local_width-2;
    // Return at everywhere in the tile, even the halos
    auto at_tile = [&] (const uint64_t row, const uint64_t& col) {
    	// Halo offset for every row>0
        // Column offset for the current row
        // Offset for row after halo
        assert(row != 0 || col < halo); // Out of bounds above
        assert(row != local_height-1 || col < halo); // Out of bounds below
        return (row==0) + col + row*local_width + (row==local_height-1);
    };
    while (error > 1E-4 && counter < 1UL << 14) {

        MPI::Request req[8];
		// There is a neighbour below so send it some juicy data
        if (verti+1 < radix) {
        	// Do *not* send the halo in the row, but only the data
            req[0] = MPI::COMM_WORLD.Isend(&ying[at_tile(local_height-2, 1)],
            	local_width-2, MPI::DOUBLE, rank+radix, 0);
            // Receive at halo
            req[1] = MPI::COMM_WORLD.Irecv(&ying[at_tile(local_height-1, 0)],
            	local_width-2, MPI::DOUBLE, rank+radix, 0);
        }
		// Oh, there is a tile above us. Let's give him/her/it some data
        if (verti > 0) {
        	// Send data
            req[2] = MPI::COMM_WORLD.Isend(&ying[at_tile(1, 1)], 
            	local_width-2, MPI::DOUBLE, rank-radix, 0);
            // Receive in halo
            req[3] = MPI::COMM_WORLD.Irecv(&ying[at_tile(0,0)], 
            	local_width-2, MPI::DOUBLE, rank-radix, 0);
        }
		// Oh my!!! There is a neighbour right of us.
        if (horiz+1 < radix) {
        	// Send data right of us
            req[4] = MPI::COMM_WORLD.Isend(&ying[at_tile(1, local_width-2)],
            	1, col_t, rank+1, 0);
            req[5] = MPI::COMM_WORLD.Irecv(&ying[at_tile(1, local_width-1)],
            	1, col_t, rank+1, 0);
        }
		// You know...
        if (horiz > 0) {
            req[6] = MPI::COMM_WORLD.Isend(&ying[at_tile(1, 1)], 
            	1, col_t, rank-1, 0);
            req[7] = MPI::COMM_WORLD.Irecv(&ying[at_tile(1, 0)],
            	1, col_t, rank-1, 0); 
        }

        // relax every pixel in the interior
        for (uint64_t row = 2; row < local_height-2; ++row)
            for (uint64_t col = 2; col < local_width-2; ++col)
                yang[local_at(row, col)] = update(row, col);

        // Wait until data received
        req[1].Wait(); req[3].Wait(); req[5].Wait(); req[7].Wait();

        // fix the borders
        for (uint64_t col = 2; col < local_width-2; ++col)
            yang[local_at(1, col)] = update(1, col);
        for (uint64_t col = 2; col < local_width-2; ++col)
            yang[local_at(local_height-2, col)] = update(local_height-2, col);
        for (uint64_t row = 1; row < local_height-1; ++row)
            yang[local_at(row, 1)] = update(row, 1);
        for (uint64_t row = 1; row < local_height-1; ++row)
            yang[local_at(row, local_width-2)] = update(row, local_width-2);

        // Wait until data send
        req[0].Wait(); req[2].Wait(); req[4].Wait(); req[6].Wait();

        double local_error = 0;
        for (uint64_t row = 1; row < local_height-1; ++row) {
            for (uint64_t col = 1; col < local_width-1; ++col) {
                    const double residue = ying[local_at(row,col)]
                                         - yang[local_at(row, col)];
                    local_error += residue*residue;
                    ying[local_at(row,col)] = yang[local_at(row, col)];
            }
        }

        // every process needs the same error
        MPI::COMM_WORLD.Allreduce(&local_error, &error, 1,
                                  MPI::DOUBLE, MPI::SUM);

        // status updates every print_every iteration
        if (counter++ % print_every == print_every-1 && is_root)
            std::cout << "# Squared error after " << counter
                      << " iterations: " << error << std::endl;
    }

    // final status
    if (is_root)
        std::cout << "# Final squared error after " << counter
                  << " iterations: " << error << std::endl;

    //////////////////////////////////////////////////////////////////////////
    // GATHER
    //////////////////////////////////////////////////////////////////////////
    // create tile data type
    MPI::Datatype tile_t =
        MPI::DOUBLE.Create_vector (local_height, local_width , width)
                   .Create_resized(0, sizeof(double));
    tile_t.Commit();
    int32_t counts[num_ranks], displs[num_ranks];
    for (uint64_t proc = 0; proc < num_ranks; ++proc) {
        const uint64_t i = proc / radix, j = proc % radix;
        counts[proc] = 1;
        displs[proc] = i*(local_height-2)*width+j*(local_width-2);
    }
    MPI::COMM_WORLD.Gatherv( ying, local_height*local_width,MPI::DOUBLE,
        image, counts, displs, tile_t, 0);

    //////////////////////////////////////////////////////////////////////////
    // CHECK
    //////////////////////////////////////////////////////////////////////////

    if (is_root) {
        std::string  filename = random_string(8)+".bmp";
        dump_bitmap(image, height, width, filename);
        std::cout << "# See "
                  << filename << "\nParallel programming is "
                  << (counter == 8353 ? "fun!" : "error-prone!") << std::endl;
    }
    
    SUCCESS_OR_DIE( gaspi_wait ( queue_id, GASPI_BLOCK ) );
    SUCCESS_OR_DIE( gaspi_proc_term(GASPI_BLOCK) );
  
    MPI::Finalize();

}