#include "gtest_lite.hpp"

#include "riscv_cv.hpp"
#include "reference_cv.hpp"
#include "test_utils.hpp"

#include <tuple>

class RGBToGrayTest : public ::testing::TestWithParam<std::tuple<int, int>>
{
};

TEST_P(RGBToGrayTest, CompareWithReference)
{
    const auto [width, height] = GetParam();

    printf("Testing RGBToGray: width=%d, height=%d\n",
           width, height);

    // RGB input image
    Image<uint8_t, ImageType::RGB> input(width, height);

    // Grayscale outputs
    Image<uint8_t, ImageType::GRAY> expected(width, height);
    Image<uint8_t, ImageType::GRAY> actual(width, height);

    // Deterministic random generator
    RandomInt<uint8_t> random(0, 255);

    // Fill ALL RGB channels.
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            input.SetPixel(x, y, random.get(), 0); // R
            input.SetPixel(x, y, random.get(), 1); // G
            input.SetPixel(x, y, random.get(), 2); // B
        }
    }

    // Golden scalar implementation
    ref::RGBToGray(input, expected);

    // RVV implementation
    vec::RGBToGray(input, actual);

    // Compare pixel by pixel
    ExpectImagesEqual(expected, actual);
}

INSTANTIATE_TEST_SUITE_P(
    RGBToGraySizes,
    RGBToGrayTest,
    ::testing::Values(
        std::make_tuple(1, 1),
        std::make_tuple(7, 1),
        std::make_tuple(16, 1),
        std::make_tuple(17, 1),
        std::make_tuple(31, 13),
        std::make_tuple(32, 13),
        std::make_tuple(33, 13),
        std::make_tuple(47, 22),
        std::make_tuple(48, 22)
    )
);