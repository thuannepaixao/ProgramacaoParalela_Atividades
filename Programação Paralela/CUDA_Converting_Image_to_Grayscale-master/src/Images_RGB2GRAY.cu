/************************* Converting an Image to Grayscale using the GPU********************/
// Grayscale Wikipedia
// Ref: https://github.com/Ohjurot/CUDATutorial
// Ref: https://www.youtube.com/watch?v=sltSyddAGNs
/*REf: https://en.wikipedia.org/wiki/Grayscale#:~:text=or%2032%20bits.-,Converting%20color%20to%20grayscale,photographic%20filters%20on%20the%20cameras.*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<iostream>


// Load Image and Write Image

#include"../includes/stb_image.h"
#include"../includes/stb_image_write.h"
#include <string>
#include <cassert>

/*
Color to Grayscale Equation
Ylinear = 0.2126RLinear + 0.7152GLiners +0.0722BLiners
*/


/*
To Load Images in C++
STB Libraris
Ref: https://github.com/nothings/stb
    Image loader: stb_image.h
    Image writer: stb_image_write.h

    folder: C:\Users\ansor\source\repos\stb
*/

// File name:

//std::string imagefilename = "C:/Users/ansor/source/repos/CUDA_Converting_Image_to_Grayscale/images/apple.jpg";

/* Define Pixel Struct */
// An output image with N components has the following components interleaved
// in this order in each pixel:
//
//     N=#comp     components
//       1           grey
//       2           grey, alpha
//       3           red, green, blue
//       4           red, green, blue, alpha
struct Pixel {
    unsigned char red, green, blue, alpha;
};


// Convert Image to Gray at CPU
void ConvertImageToGrayCpu(unsigned char* imageRGBA, int width, int height) {
    for (int y = 0;y < height;y++) {
        for (int x = 0; x < width;x++) {

            /*  Color to Grayscale Equation
             Ylinear = 0.2126RLinear + 0.7152GLiners +0.0722BLiners */
            Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
            unsigned char pixelValue = (unsigned char) (ptrPixel->red * 0.2126f + ptrPixel->green * 0.7152f + ptrPixel->blue * 0.0722f);

            //float pixelValue = ptrPixel->red * 0.2126f + ptrPixel->green * 0.7152f + ptrPixel-> blue * 0.0722f;
            //unsigned char pixelValue = pixelValue;
            ptrPixel->red = pixelValue;
            ptrPixel->green = pixelValue;
            ptrPixel->blue = pixelValue;
            ptrPixel->alpha = 255;

        }
    }

}

// Convert Image to Gray at GPU

__global__ void ConvertImageToGrayGPU(unsigned char* imageRGBA) {

    uint32_t x      = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y      = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idx    = y * blockDim.x * gridDim.x + x;

    /*  Color to Grayscale Equation
      Ylinear = 0.2126RLinear + 0.7152GLiners +0.0722BLiners */
    Pixel* ptrPixel = (Pixel*)&imageRGBA[idx * 4];
    unsigned char pixelValue = (unsigned char)
        (ptrPixel->red * 0.2126f + ptrPixel->green * 0.7152f + ptrPixel->blue * 0.0722f);

    //float pixelValue = ptrPixel->red * 0.2126f + ptrPixel->green * 0.7152f + ptrPixel-> blue * 0.0722f;
    //unsigned char pixelValue = pixelValue;
    ptrPixel->red = pixelValue;
    ptrPixel->green = pixelValue;
    ptrPixel->blue = pixelValue;
    ptrPixel->alpha = 255;


}


//
int main(int argc , char** argv) {

    // Check Argument count
    if (argc < 2) {
        std::cout << "Usage: 02_ImageToGray <filename1>" << std::endl;
        return -1;
    }

    // open file 
    /* 
    // Basic usage (see HDR discussion below for HDR usage):
    //    int x,y,n;
    //    unsigned char *data = stbi_load(filename, &x, &y, &n, 0);
    //    // ... process data if not NULL ...
    //    // ... x = width, y = height, n = # 8-bit components per pixel ...
    //    // ... replace '0' with '1'..'4' to force that many components per pixel
    //    // ... but 'n' will always be the number that it would have been if you said 0
    //    stbi_image_free(data)
    */

    // Open Image
    int width, height, componentCount;
    int width1, height1, componentCount1;
    //auto filename = "apple.jpg";
    auto filename = "ship_4k_rgba.png";

   // char const* filepath = "../images/apple.jpg";

    unsigned char* imageData = stbi_load(filename, &width, &height, &componentCount, 4);
    // add filename at debug config
    unsigned char* imageData1 = stbi_load(argv[1], &width1, &height1, &componentCount1, 4);

    if (!imageData || !imageData1) {
        std::cout << "Failed to Open: "<<filename << std::endl;
        std::cout << "Failed to Open: \"" <<argv[1] << "\"";
    }
    else {
        std::cout << "File Open OK: " << typeid(filename).name() << std::endl;
        std::cout << "File Open argv[1] OK: " << typeid(argv[1]).name() << std::endl;
 
        std::cout << "imageData Type: " << typeid(&imageData).name() << std::endl;
        std::cout << "File Open OK: " << filename << std::endl;
        std::cout << "width: " << width << std::endl;
        std::cout << "height: " << height << std::endl;
        std::cout << "componentCount: " << componentCount << std::endl;

        std::cout << "File Open OK: " << argv[1] << std::endl;
        std::cout << "width: " << width1 << std::endl;
        std::cout << "height: " << height1 << std::endl;
        std::cout << "componentCount: " << componentCount1 << std::endl;
    }

    //Validate Image Sizes
    //Compute Images on the 32 by 32 Pixel Blocks base = 1024 which is the maximum size
    
    // Check Image 1
    // Check Our Image is going to be dividable by 32
    if (width % 32 || height % 32) {

        //Note:Leaked memory of "ImageData
        std::cout << " Width and/or Height is not dividable by 32!!"<<std::endl;
        return -1;
    }
    else {
        std::cout << " OK: Images Dividable by 32" << std::endl;
    }
    // Check Image 2
    // Check Our Image is going to be dividable by 32
    if (width1 % 32 || height1 % 32) {

        //Note:Leaked memory of "ImageData
        std::cout << " Width and/or Height is not dividable by 32!!" << std::endl;
        return -1;
    }
    else {
        std::cout << " OK: Images Dividable by 32" << std::endl;
    }

    // TODO: Process Image

    /*
    // Processing Image on CPU 
    std::cout << "Processing Images..." << std::endl;
    // Call functions CPU
    ConvertImageToGrayCpu(imageData, width, height);
     std::cout << "DONE " << std::endl;
     */

     // Process Image on GPU
    // Copy data to GPU
    std::cout << "Copy Data to GPU..." << std::endl;
    unsigned char* ptrImageDataGPU = nullptr;
    assert(cudaMalloc(&ptrImageDataGPU, width * height * 4)==cudaSuccess);
    assert(cudaMemcpy(ptrImageDataGPU, imageData, width * height * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    std::cout << "DONE " << std::endl;

    std::cout << "Processing Images..." << std::endl;
    std::cout << " Running CUDA Kernel ..." << std::endl;
    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);

    ConvertImageToGrayGPU << <gridSize , blockSize >> > (ptrImageDataGPU);

     std::cout << "DONE " << std::endl;

     // Copy data from the GPU to CPU
     std::cout << " Copy Data from GPU to CPU.." << std::endl;
     assert(cudaMemcpy(imageData,ptrImageDataGPU , width* height *4, cudaMemcpyDeviceToHost) == cudaSuccess);
     std::cout << "DONE " << std::endl;


    // Build output filename
    std::string fileNameOut = filename;
    std::string fileNameOut1 = argv[1];

    fileNameOut = fileNameOut.substr(0, fileNameOut.find_last_of(".")) + "_gray.png";
    fileNameOut1 = fileNameOut1.substr(0, fileNameOut1.find_last_of(".")) + "_gray.png";

    std::cout << fileNameOut << std::endl;
    std::cout << fileNameOut1 << std::endl;

    // White image back to disk
    stbi_write_png(fileNameOut.c_str(),width, height,4, imageData, 4 *width);
    stbi_write_png(fileNameOut1.c_str(), width1, height1, 4, imageData1, 4 * width1);
    std::cout << " White image back to disk: OK" << std::endl;
    std::cout <<"DONE" << std::endl;


    //Free memory-  Close Image
    cudaFree(ptrImageDataGPU);
    stbi_image_free(imageData);
    stbi_image_free(imageData1);


}