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
    // TODO argv params for SE and block size

    std::vector<std::string> filename = {"logitech_bill_clinton_bin.ppm", "micropro_wordstar_bin.ppm"};

    std::vector<double> times = new std::vector<double>(filename.size() * 6);
    Image_t *input_img, *img, *output;
    std::string name;
    std::chrono::high_resolution_clock::time_point t_start, t_end;
    std::chrono::duration<double> time_span;

    for(auto file = filename.begin(); file != filename.end(); ++file){
        input_img = PPM_import(file->c_str());
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
		output = erosion(img, se, &time_span);
        cudaDeviceSynchronize();
		times.push_back(time_span.qualcosa()) // FIXME
		PPM_export((file + "_eroded.ppm").c_str(), output);
		Image_delete(output);

		// DILATATION
        output = dilatation(img, se, &time_span);
        cudaDeviceSynchronize();
        times.push_back(time_span.qualcosa()) // FIXME
        PPM_export((file + "dilatated.ppm").c_str(), output);
        Image_delete(output);

        // OPENING
        output = opening(img, se, &time_span);
        cudaDeviceSynchronize();
        times.push_back(time_span.qualcosa()) // FIXME
        PPM_export((file + "_opened.ppm").c_str(), output);
        Image_delete(output);

        // CLOSING
        output = closing(img, se, &time_span);
        cudaDeviceSynchronize();
        times.push_back(time_span.qualcosa()) // FIXME
        PPM_export((file + "_closed.ppm").c_str(), output);
        Image_delete(output);

        // TOPHAT
        output = topHat(img, se, &time_span);
        cudaDeviceSynchronize();
        times.push_back(time_span.qualcosa()) // FIXME
        PPM_export((file + "_topHat.ppm").c_str(), output);
        Image_delete(output);

        // BOTTOMHAT
        output = bottomHat(img, se, &time_span);
        cudaDeviceSynchronize();
        times.push_back(time_span.qualcosa()) // FIXME
        PPM_export((file + "_bottomHat.ppm").c_str(), output);
        Image_delete(output);

        free(input);
    }

    fstream timings_file = NULL; // FIXME
    int i = 0;
    for(auto file = filename.begin(); file != filename.end(); ++file){
        timings_file << file.c_str() << endl;
        timings_file << "EROSION;" <<times[i++] <<endl;
        timings_file << "DILATATION;" <<times[i++] <<endl;
        timings_file << "OPENING;" <<times[i++] <<endl;
        timings_file << "CLOSING;" <<times[i++] <<endl;
        timings_file << "TOPHAT;" <<times[i++] <<endl;
        timings_file << "BOTTOMHAT;" <<times[i++] <<endl;
    }
    // FIXME close file

	return 0;
}
