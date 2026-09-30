#pragma once

#include "image.hpp"

namespace ref
{
    void RGBToGray(const Image<uint8_t, ImageType::RGB>& input,
                         Image<uint8_t, ImageType::GRAY>& output);
}