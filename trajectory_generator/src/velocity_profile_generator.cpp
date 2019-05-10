#include <iostream>
#include <random>

int test (std::vector<double> &result, int total_frames, int treshold_frame, double total_length, double treshold_length)
{
    std::cout << "test function" << std::endl;
    double slope_end_boundaries[] = {0.1, 1}; // limit in percentage of the total_frames for the end of the slope in the velocity profile
    std::default_random_engine generator {std::random_device()()};
    std::uniform_int_distribution<int> distribution(treshold_frame*slope_end_boundaries[0], treshold_frame*slope_end_boundaries[1]);

    std::vector<int> dist;
    for (int i = 0; i < 100; i++)
    {
        dist.push_back(0);
    }
    dist[distribution(generator)] += 1;

    for (unsigned i = 0; i < dist.size(); i++)
    {
        std::cout << "dist[" << i << "] = " << dist[i] << std::endl;
    }
}