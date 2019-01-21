#include <iostream>
#include <cstdint>
#include <vector>
#include <limits>
#include <string>
#include <algorithm>

#include <assert.h>
#include <mpi.h>

#include "include/bitmap_IO.hpp"

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

    MPI::Init (argc, argv);

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

    const uint64_t local_height = (height-2)/radix+2;
    const uint64_t local_width  = (width -2)/radix+2;
    std::vector<double> image(height*width*is_root);
    std::vector<double> ying(local_height*local_width, 0);
    std::vector<double> yang(local_height*local_width, 0);

    // make indexing great again!
    auto       at = [&] (const uint64_t& row, const uint64_t& col) {
        return row*width+col;
    };
    auto local_at = [&] (const uint64_t& row, const uint64_t& col) {
        return row*local_width+col;
    };

    if (is_root) {
        // draw a checkerboard
        const uint64_t stride = 181;
        for (uint64_t row = 0; row < height; ++row)
            for (uint64_t col = 0; col < width; ++col)
                image[at(row, col)] = (row/stride + col/stride) % 2;
    }

    //////////////////////////////////////////////////////////////////////////
    // SCATTER
    //////////////////////////////////////////////////////////////////////////




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

       
        // Send the data

        // relax every pixel in the interior
        for (uint64_t row = 2; row < local_height-2; ++row)
            for (uint64_t col = 2; col < local_width-2; ++col)
                yang[local_at(row, col)] = update(row, col);

        // Wait until data received

            
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

        
        double local_error = 0;
        for (uint64_t row = 1; row < local_height-1; ++row) {
            for (uint64_t col = 1; col < local_width-1; ++col) {
                    const double residue = ying[local_at(row,col)]
                                         - yang[local_at(row, col)];
                    local_error += residue*residue;
                    ying[local_at(row,col)] = yang[local_at(row, col)];
            }
        }

    
    

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

    
                  
    //////////////////////////////////////////////////////////////////////////
    // CHECK
    //////////////////////////////////////////////////////////////////////////

    if (is_root) {
        std::string  filename = random_string(8)+".bmp";
        dump_bitmap(image.data(), height, width, "www/"+filename);
        std::cout << "# See http://iaimz105.informatik.uni-mainz.de/"
                  << filename << "\nParallel programming is "
                  << (counter == 8353 ? "fun!" : "error-prone!") << std::endl;
    }

    MPI::Finalize();

}