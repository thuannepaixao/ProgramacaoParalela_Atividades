#include <iostream>
#include <fstream>

//Load Images
#define STB_IMAGE_IMPLEMENTATION
// Write Images
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "include/stb_image.h"
//#include "stb/stb_image_resize.h"
#include "include/stb_image_write.h"


using namespace std;

int main()
{
    int width, height, channels;
    //loading jpeg file
    unsigned char* img = stbi_load("images/apple.jpg", &width, &height, &channels, 4);
    // Check If image was loads correct!!
    if (img == 0) {
        cout << "Error loading image file" << endl;
        return -1;
    }

    cout << "Loading images\n";
    cout << "\twidth = " << width << "\n";
    cout << "\theight = " << height << "\n";
    cout << "\tchannels = " << channels << "\n";

    //writing jpeg images

    stbi_write_jpg("apple_copy01.jpg", width, height, channels, img, 100);
    stbi_image_free(img);
    return 0;

}