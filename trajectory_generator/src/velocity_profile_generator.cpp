#include <iostream>
#include <random>
#include <cmath>

int generate_velocity_profile (std::vector<double> &result, int total_frames, int treshold_frame, double total_length, double treshold_length)
{
    int slope_end_frame;
    double final_part_ds;
    double velocity_upper_limit;
    double slope_end_boundaries[] = {0.5, 1}; // limit in percentage of the total_frames for the end of the slope in the velocity profile
    double computed_total_distance;
    std::default_random_engine generator {std::random_device()()};
    std::uniform_int_distribution<int> distribution(treshold_frame*slope_end_boundaries[0], treshold_frame*slope_end_boundaries[1]);

    // ===============================================
    double release_velocity;
    int starting_frame;
    // std::uniform_real_distribution<> real_distribution(2 * treshold_length / treshold_frame, treshold_length / 4); 
    std::uniform_real_distribution<> real_distribution(2 * treshold_length / treshold_frame, 3 * treshold_length / treshold_frame);
    // std::uniform_real_distribution<> real_distribution(2 * treshold_length / treshold_frame, 3 * treshold_length / treshold_frame);

    // std::uniform_real_distribution<> real_distribution(2 * treshold_length / treshold_frame, 4 * treshold_length / treshold_frame);

    release_velocity = real_distribution(generator);
    starting_frame = treshold_frame - (2 * treshold_length / release_velocity);

    computed_total_distance = 0;
    for (unsigned i = 0; i < starting_frame; i++)
    {
        result[i] = 0;
        computed_total_distance += result[i];
    }
    for (unsigned i = starting_frame; i < treshold_frame; i++)
    {
        result[i] = static_cast<double>(i + 1 - starting_frame) / (treshold_frame - starting_frame) * release_velocity;
        computed_total_distance += result[i];
        // std::cout << i << std::endl;
        // std::cout << "result[" << i << "] = " << result[i] << std::endl;
    }
    for (unsigned i = treshold_frame; i < total_frames; i++)
    {
        // result[i] = result[treshold_frame-1] * (total_frames - (i+1)) / (total_frames - treshold_frame);

        result[i] = result[treshold_frame-1];
        // result[i] = result[i-1] * (total_frames - i) / (total_frames - treshold_frame);
        // result[i] = static_cast<double>(release_velocity * (total_frames - (i+1)) / (total_frames - treshold_frame));
        // result[i] = result[i-1] * 0.7;
        computed_total_distance += result[i];
        // if (computed_total_distance > total_length)
        // {
        //     std::cout << "CUT" << std::endl;
        //     result[i] = 0;
        //     for (; i < total_frames; i++)
        //     {
        //         result[i] = 0;
        //     }
        // }
    }
    return 1;
    
    // ===============================================
    // slope_end_frame = distribution(generator);
    // // when slope_end_frame is an odd number the total length computation is better
    // if (slope_end_frame % 2 == 0)
    // {
    //     if (treshold_frame*slope_end_boundaries[1] - slope_end_frame > slope_end_frame - treshold_frame*slope_end_boundaries[0])
    //     {
    //         slope_end_frame++;
    //     }
    //     else
    //     {
    //         slope_end_frame--;
    //     }
    // }

    // final_part_ds = (total_length - treshold_length) / (total_frames - treshold_frame);
    // velocity_upper_limit = treshold_length / (treshold_frame - (slope_end_frame / 2));

    // computed_total_distance = 0;
    // std::cout << "velocity_upper_limit = " << velocity_upper_limit << std::endl;
    // for (unsigned i = 0; i < slope_end_frame; i++)
    // {
    //     result[i] = (static_cast<double>(i+1) / slope_end_frame) * velocity_upper_limit;
    //     computed_total_distance += result[i];
    // }
    // for (unsigned i = slope_end_frame; i < treshold_frame; i++)
    // {
    //     result[i] = velocity_upper_limit;
    //     computed_total_distance += result[i];
    // }
    // for (unsigned i = treshold_frame; i < total_frames; i++)
    // {
    //     // result[i] = final_part_ds;
    //     // result[i] = ((static_cast<double>(total_frames - (i+1)) / static_cast<double>(total_frames - treshold_frame)) * velocity_upper_limit);
    //     // result[i] = ((static_cast<double>(total_frames - (i+1)) / static_cast<double>(total_frames - treshold_frame)) * (final_part_ds * 2));      
    //     result[i] = velocity_upper_limit * (total_frames - i - 1) / (total_frames - treshold_frame);
    //     computed_total_distance += result[i];
    //     if (computed_total_distance > total_length)
    //     {
    //         result[i] = 0;
    //         for (; i < total_frames; i++)
    //         {
    //             result[i] = 0;
    //         }
    //     }
    // }

    // return 1;
}

int decelerate_joints()
{
    return 37;
}