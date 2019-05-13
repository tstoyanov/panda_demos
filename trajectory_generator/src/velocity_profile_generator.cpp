#include <iostream>
#include <random>
#include <cmath>

int velocity_profile_generator (std::vector<double> &result, int total_frames, int treshold_frame, double total_length, double treshold_length)
{
    std::cout << "test function" << std::endl;
    int slope_end_frame;
    double final_part_ds;
    double velocity_upper_limit;
    double slope_end_boundaries[] = {0.5, 0.9}; // limit in percentage of the total_frames for the end of the slope in the velocity profile
    std::default_random_engine generator {std::random_device()()};
    std::uniform_int_distribution<int> distribution(treshold_frame*slope_end_boundaries[0], treshold_frame*slope_end_boundaries[1]);

    slope_end_frame = distribution(generator);
    // when slope_end_frame is an odd number the total length computation is better
    if (slope_end_frame % 2 == 0)
    {
        if (treshold_frame*slope_end_boundaries[1] - slope_end_frame > slope_end_frame - treshold_frame*slope_end_boundaries[0])
        {
            slope_end_frame++;
        }
        else
        {
            slope_end_frame--;
        }
    }

    final_part_ds = (total_length - treshold_length) / (total_frames - treshold_frame);
    velocity_upper_limit = treshold_length / (treshold_frame - (slope_end_frame / 2));

    std::cout << "velocity_upper_limit = " << velocity_upper_limit << std::endl;
    for (unsigned i = 0; i < slope_end_frame; i++)
    {
        result[i] = (static_cast<double>(i+1) / slope_end_frame) * velocity_upper_limit;
    }
    for (unsigned i = slope_end_frame; i < treshold_frame; i++)
    {
        result[i] = velocity_upper_limit;
    }
    for (unsigned i = treshold_frame; i < total_frames; i++)
    {
        // result[i] = final_part_ds;
        result[i] = ((static_cast<double>(total_frames - (i+1)) / static_cast<double>(total_frames - treshold_frame)) * (final_part_ds * 2));
    }

    return 1;
}