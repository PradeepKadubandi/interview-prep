#pragma once
#include "network.h"
#include "math.h"
using namespace std;

#define PI 3.14159265
#define EARTH_RADIUS 6356.752

/*
Reference: https://en.wikipedia.org/wiki/Great-circle_distance
Used the simple formula without bothering about rounding errors
*/
double calculateGreatCircleDistance(row r1, row r2) {
    double degree_to_radians =  PI / 180.0;
    auto lon1 = r1.lon * degree_to_radians;
    auto lon2 = r2.lon * degree_to_radians;
    auto lat1 = r1.lat * degree_to_radians;
    auto lat2 = r2.lat * degree_to_radians;
    auto delta_lon = lon2 - lon1;
    auto angle = acos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(delta_lon));
    return EARTH_RADIUS * angle;
};

bool is_better_time(int node_id, double curr_rem_range, double curr_time_to_reach, double new_rem_range, double new_time_to_reach) {
    auto inf = numeric_limits<double>::max();
    if (new_time_to_reach == inf) {
        return false;
    }
    if (curr_time_to_reach == inf) {
        return true;
    }
    auto rate = network[node_id].rate;
    if (new_rem_range > curr_rem_range) {
        return ((new_rem_range - curr_rem_range) / rate) > (new_time_to_reach - curr_time_to_reach);
    }
    else {
        return ((curr_rem_range - new_rem_range) / rate) > (curr_time_to_reach - new_time_to_reach);
    }
};