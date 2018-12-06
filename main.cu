#include <cstdlib>
#include <cstdio>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <vector>
#include "Image.h"
#include "PPM.h"
#include "MorphableOperator.h"

#define SE_W 5
#define SE_H 5

int main (int argc, char **argv){

    std::vector<std::string> filename = {"logitech_bill_clinton_bin.ppm", "micropro_wordstar_bin.ppm"};
    /*
    filename.push_back("logitech_bill_clinton_bin.ppm");
    filename.push_back("micropro_wordstar_bin.ppm");
    filename.push_back("apple_adam_bin.ppm");
    filename.push_back("two_bytes_better_bin.ppm"); //"two_bytes_better_bin.ppm");*/

    double times[filename.size() * 6];
    int times_index = 0;
    Image_t *input_img, *img, *out;
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
		Image_t* output = erosion(img, se, &time_span);



		cudaDeviceSynchronize();
    }





	return 0;
}
