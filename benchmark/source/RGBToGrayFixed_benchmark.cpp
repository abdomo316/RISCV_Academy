#include "benchmark.hpp"
#include "reference_cv.hpp"

int main()
{
#ifndef RISCV_QEMU
    riscv_enable_vector();
#endif

    constexpr int width = 128;
    constexpr int height = 100;
    constexpr int loop_count = 3;

    Image<uint8_t, ImageType::RGB> input(width, height);

    Image<uint8_t, ImageType::GRAY> reference_output(width, height);
    Image<uint8_t, ImageType::GRAY> vector_output(width, height);

    RandomInt<uint8_t> random(0, 255);

    // Initialize RGB input.
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            input.SetPixel(x, y, random.get(), 0);
            input.SetPixel(x, y, random.get(), 1);
            input.SetPixel(x, y, random.get(), 2);
        }
    }

    uint64_t scalar_cycles = 0;
    uint64_t scalar_instrs = 0;

    uint64_t vector_cycles = 0;
    uint64_t vector_instrs = 0;

    for (int i = 0; i < loop_count; ++i)
    {
        // Scalar fixed-point implementation.
        Timer scalar_timer;

        scalar_timer.Start();

        ref::RGBToGrayFixed(
            input,
            reference_output);

        scalar_timer.Stop();

        scalar_cycles += scalar_timer.ElapsedCycles();
        scalar_instrs += scalar_timer.ElapsedInstructions();

        // RVV fixed-point implementation.
        Timer vector_timer;

        vector_timer.Start();

        vec::RGBToGrayFixed(
            input,
            vector_output);

        vector_timer.Stop();

        vector_cycles += vector_timer.ElapsedCycles();
        vector_instrs += vector_timer.ElapsedInstructions();
    }

    scalar_cycles /= loop_count;
    scalar_instrs /= loop_count;

    vector_cycles /= loop_count;
    vector_instrs /= loop_count;

    const bool correct =
        CheckCorrectness(
            reference_output,
            vector_output);

    PrintTime(
        "RGBToGrayFixed",
        width * height,
        vector_cycles,
        vector_instrs,
        scalar_cycles,
        scalar_instrs);

    if (correct)
    {
        printf("Output is correct.\n");
        return 0;
    }

    printf("Output is wrong.\n");
    return 1;
}