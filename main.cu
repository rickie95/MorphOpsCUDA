#include <cstdlib>
#include <cstdio>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <vector>
#include "Image.h"
#include "PPM.h"
#include "MorphableOperator.h"

int main (int argc, char **argv){

	std::string path_input = "/home/user/myfolder/";
	std::string path_results = path_input;

    std::vector<std::string> filename = {"myFileA.ppm", "myFileB.ppm", "myFileC.ppm", "myFileD.ppm"};

    std::vector<double> times[filename.size() * 6];
    Image_t *input_img, *img, *output;
    std::string name;
    std::chrono::high_resolution_clock::time_point t_start, t_end;
    std::chrono::duration<double> time_span;

    for(auto file = filename.begin(); file != filename.end(); ++file){
        input_img = PPM_import((path_input + *file).c_str());
        printf("\nLoaded %s (%dx%d) \n", file->c_str(), input_img->width, input_img->height);

        // 0.0 -> BLACK ; 1.0 -> WHITE

        StructElem* se = new DiamondShape_SE(3);

        // Extract only the first channel.
		float *input = input_img->data;
		if(input_img->channels > 1) {
			input = (float *) malloc(input_img->width * input_img->height * sizeof(float));
			for (int r = 0; r < input_img->height; r += 1) {
				for (int c = 0; c < input_img->width; c += 1) {
					input[r * input_img->width + c] = input_img->data[r * input_img->width * 3 + c * 3];
				}
			}
		}

        img = Image_new(input_img->width, input_img->height, 1, input);

		// ELABORATION STAGE
		// EROSION
        printf("Erosion...\t");
		output = erosion(img, se, &time_span);
        cudaDeviceSynchronize();
		times->push_back(time_span.count());
		PPM_export((*file +(std::string)"_eroded.ppm").c_str(), output);
		Image_delete(output);


		// DILATATION
		printf("Dilatation...\t");
        output = dilatation(img, se, &time_span);
        cudaDeviceSynchronize();
        times->push_back(time_span.count());
        PPM_export((path_results + *file +(std::string)"dilatated.ppm").c_str(), output);
        Image_delete(output);

        // OPENING
        printf("Opening...\t");
        output = opening(img, se, &time_span);
        cudaDeviceSynchronize();
        times->push_back(time_span.count());
        PPM_export((path_results + *file +(std::string)"_opened.ppm").c_str(), output);
        Image_delete(output);

        // CLOSING
        printf("Closing...\t");
        output = closing(img, se, &time_span);
        cudaDeviceSynchronize();
        times->push_back(time_span.count());
        PPM_export((path_results + *file +(std::string)"_closed.ppm").c_str(), output);
        Image_delete(output);

        // TOPHAT
        printf("TOPHAT...\t");
        output = topHat(img, se, &time_span);
        cudaDeviceSynchronize();
        times->push_back(time_span.count());
        PPM_export((path_results + *file +(std::string)"_topHat.ppm").c_str(), output);
        Image_delete(output);

        // BOTTOMHAT
        printf("BOTTOM HAT... \n");
        output = bottomHat(img, se, &time_span);
        cudaDeviceSynchronize();
        times->push_back(time_span.count());
        PPM_export((path_results + *file +(std::string)"_bottomHat.ppm").c_str(), output);
        Image_delete(output);

        free(input);
    }
    printf("Writing times on file...\n");
    std::ofstream timings_file;
    std::string fname = "timings_"+ std::to_string(TILE_WIDTH) + ".csv";
    timings_file.open((path_results + fname).c_str());
    auto it = times->begin();
    for(auto file = filename.begin(); file != filename.end(); ++file){
        timings_file << file->c_str() << "(TILE_WIDTH="<<TILE_WIDTH<<")"<< "\n";
        timings_file << "EROSION;" << *it++ << "\n";
        timings_file << "DILATATION;" <<*it++<<"\n";
        timings_file << "OPENING;" << *it++ <<"\n";
        timings_file << "CLOSING;" << *it++ <<"\n";
        timings_file << "TOPHAT;" << *it++ <<"\n";
        timings_file << "BOTTOMHAT;" << *it++ <<"\n";
    }
    timings_file.close();
    printf("==== DONE ==== \n");
	return 0;
}
