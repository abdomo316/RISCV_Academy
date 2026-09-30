#include "image_RGBToGray.hpp"

#include <cstddef>
#include <riscv_vector.h>

void vec::RGBToGray(const Image<uint8_t, ImageType::RGB>& input,
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
            // Number of pixels processed in this iteration.
            vl = __riscv_vsetvl_e8m2(width - x);

            // RGB layout:
            // R0 G0 B0 R1 G1 B1 R2 G2 B2 ...
            const ptrdiff_t stride = 3;

            // Load R, G and B channels separately.
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

            // uint8 -> uint16.
            const vuint16m4_t r16 =
                __riscv_vwaddu_vx_u16m4(r, 0, vl);

            const vuint16m4_t g16 =
                __riscv_vwaddu_vx_u16m4(g, 0, vl);

            const vuint16m4_t b16 =
                __riscv_vwaddu_vx_u16m4(b, 0, vl);

            // uint16 -> uint32.
            const vuint32m8_t r32 =
                __riscv_vwaddu_vx_u32m8(r16, 0, vl);

            const vuint32m8_t g32 =
                __riscv_vwaddu_vx_u32m8(g16, 0, vl);

            const vuint32m8_t b32 =
                __riscv_vwaddu_vx_u32m8(b16, 0, vl);

            // uint32 -> float32.
            const vfloat32m8_t rf =
                __riscv_vfcvt_f_xu_v_f32m8(r32, vl);

            const vfloat32m8_t gf =
                __riscv_vfcvt_f_xu_v_f32m8(g32, vl);

            const vfloat32m8_t bf =
                __riscv_vfcvt_f_xu_v_f32m8(b32, vl);

            // Gray = 0.299R + 0.587G + 0.114B
            vfloat32m8_t gray =
                __riscv_vfmul_vf_f32m8(
                    rf,
                    0.299f,
                    vl);

            gray =
                __riscv_vfmacc_vf_f32m8(
                    gray,
                    0.587f,
                    gf,
                    vl);

            gray =
                __riscv_vfmacc_vf_f32m8(
                    gray,
                    0.114f,
                    bf,
                    vl);

            // Round to nearest:
            // gray + 0.5, then truncate toward zero.
            gray =
                __riscv_vfadd_vf_f32m8(
                    gray,
                    0.5f,
                    vl);

            // float32 -> uint32.
            const vuint32m8_t gray32 =
                __riscv_vfcvt_rtz_xu_f_v_u32m8(
                    gray,
                    vl);

            // uint32 -> uint16.
            const vuint16m4_t gray16 =
                __riscv_vnclipu_wx_u16m4(
                    gray32,
                    0,
                    vl);

            // uint16 -> uint8.
            const vuint8m2_t gray8 =
                __riscv_vnclipu_wx_u8m2(
                    gray16,
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