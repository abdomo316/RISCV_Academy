#include "image_RGBToGray.hpp"

void ref::RGBToGray(const Image<uint8_t, ImageType::RGB>& input,
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

            const float gray =
                0.299f * static_cast<float>(r) +
                0.587f * static_cast<float>(g) +
                0.114f * static_cast<float>(b);

            // Round to nearest integer.
            output.SetPixel(
                x,
                y,
                static_cast<uint8_t>(gray + 0.5f));
        }
    }
}