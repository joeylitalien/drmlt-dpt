#include "image.h"

#include <OpenImageIO/imageio.h>

Image3::Image3(const std::string &filename) {
    std::unique_ptr<OIIO::ImageInput> in = OIIO::ImageInput::open(filename);
    if (in == nullptr) {
        Error("File not found");
    }
    const OIIO::ImageSpec &spec = in->spec();
    pixelWidth = spec.width;
    pixelHeight = spec.height;
    OIIO::TypeDesc typeDesc = sizeof(Float) == sizeof(double) ? OIIO::TypeDesc::DOUBLE
                                                                     : OIIO::TypeDesc::FLOAT;
    if (spec.nchannels == 1) {
        std::vector<Float> pixels(pixelWidth * pixelHeight);
        in->read_image(typeDesc, &pixels[0]);
        data.resize(pixelWidth * pixelHeight);
        int p = 0;
        for (int y = 0; y < pixelHeight; y++) {
            for (int x = 0; x < pixelWidth; x++) {
                Float val = pixels[p++];
                At(x, y) = Vector3(val, val, val);
            }
        }
    } else if (spec.nchannels == 3) {
        std::vector<Float> pixels(3 * pixelWidth * pixelHeight);
        in->read_image(typeDesc, &pixels[0]);
        data.resize(pixelWidth * pixelHeight);
        int p = 0;
        for (int y = 0; y < pixelHeight; y++) {
            for (int x = 0; x < pixelWidth; x++) {
                Float val0 = pixels[p++];
                Float val1 = pixels[p++];
                Float val2 = pixels[p++];
                At(x, y) = Vector3(val0, val1, val2);
            }
        }
    } else {
        Error("Unsupported number of channels");
    }
    in->close();
}

void WriteImage(const std::string &filename, const Image3 *image) {
    std::unique_ptr<OIIO::ImageOutput> out = OIIO::ImageOutput::create(filename);
    if (out == nullptr) {
        Error("Fail to create file");
        return;
    }
    OIIO::ImageSpec spec(
        image->pixelWidth, image->pixelHeight, 3, OIIO::TypeDesc::HALF);
    out->open(filename, spec);
    out->write_image(sizeof(Float) == sizeof(double) ? OIIO::TypeDesc::DOUBLE
                                                     : OIIO::TypeDesc::FLOAT,
                     &image->data[0]);
    out->close();
//    OIIO::ImageOutput::destroy(out);
}