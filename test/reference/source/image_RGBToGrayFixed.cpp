#include "image_RGBToGrayFixed.hpp"

void ref::RGBToGrayFixed(
    const Image<uint8_t, ImageType::RGB>& input,
    Image<uint8_t, ImageType::GRAY>& output)
{
    assert(input.Width() == output.Width() &&
           input.Height() == output.Height());

    for (int y = 0; y < input.Height(); ++y)
    {
        for (int x = 0; x < input.Width(); ++x)
        {
            const uint8_t r = input.GetPixel(x, y, 0);
            const uint8_t g = input.GetPixel(x, y, 1);
            const uint8_t b = input.GetPixel(x, y, 2);

            // Fixed-point approximation:
            //
            // 0.299 ≈ 77 / 256
            // 0.587 ≈ 150 / 256
            // 0.114 ≈ 29 / 256
            //
            // +128 is used for round-to-nearest.
            const uint32_t gray =
                77u  * static_cast<uint32_t>(r) +
                150u * static_cast<uint32_t>(g) +
                29u  * static_cast<uint32_t>(b) +
                128u;

            output.SetPixel(
                x,
                y,
                static_cast<uint8_t>(gray >> 8));
        }
    }
}