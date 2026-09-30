#include "image_RGBToGrayFixed.hpp"

#include <cstddef>
#include <riscv_vector.h>

void vec::RGBToGrayFixed(
    const Image<uint8_t, ImageType::RGB>& input,
    Image<uint8_t, ImageType::GRAY>& output)
{
    assert(input.Width() == output.Width() &&
           input.Height() == output.Height());

    const uint8_t* in = input.GetPtr(0, 0);
    uint8_t* out = output.GetPtr(0, 0);

    const size_t width = input.Width();
    const size_t height = input.Height();

    for (size_t y = 0; y < height; ++y)
    {
        const uint8_t* row_in = &in[y * width * 3];
        uint8_t* row_out = &out[y * width];

        size_t vl = 0;

        for (size_t x = 0; x < width; x += vl)
        {
            vl = __riscv_vsetvl_e8m2(width - x);

            // RGB input layout:
            //
            // R0 G0 B0 R1 G1 B1 R2 G2 B2 ...
            //
            // Therefore, consecutive values of the same channel
            // are separated by 3 bytes.
            const ptrdiff_t stride = 3;

            const vuint8m2_t r =
                __riscv_vlse8_v_u8m2(
                    &row_in[x * 3],
                    stride,
                    vl);

            const vuint8m2_t g =
                __riscv_vlse8_v_u8m2(
                    &row_in[x * 3 + 1],
                    stride,
                    vl);

            const vuint8m2_t b =
                __riscv_vlse8_v_u8m2(
                    &row_in[x * 3 + 2],
                    stride,
                    vl);

            // Widening multiply:
            //
            // uint8 * coefficient -> uint16
            //
            // gray = 77*R
            vuint16m4_t gray =
                __riscv_vwmulu_vx_u16m4(
                    r,
                    77,
                    vl);

            // gray += 150*G
            gray =
                __riscv_vwmaccu_vx_u16m4(
                    gray,
                    150,
                    g,
                    vl);

            // gray += 29*B
            gray =
                __riscv_vwmaccu_vx_u16m4(
                    gray,
                    29,
                    b,
                    vl);

            // Add 128 before dividing by 256
            // to implement round-to-nearest.
            gray =
                __riscv_vadd_vx_u16m4(
                    gray,
                    128,
                    vl);

            // Divide by 256 using >> 8.
            gray =
                __riscv_vsrl_vx_u16m4(
                    gray,
                    8,
                    vl);

            // Result is now guaranteed to be in [0, 255].
            const vuint8m2_t gray8 =
                __riscv_vnclipu_wx_u8m2(
                    gray,
                    0,
                    vl);

            // Store grayscale pixels.
            __riscv_vse8_v_u8m2(
                &row_out[x],
                gray8,
                vl);
        }
    }
}