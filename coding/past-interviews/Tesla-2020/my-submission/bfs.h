#pragma once
#include "network.h"
#include "utilities.h"
#include "buildGraph.h"
#include<deque>
#include<list>
using namespace std;

#define FULL_RANGE 320.0
#define SPEED 105.0

struct Metric {
    public:
        int node_id; 
        double time_from_source;
        double rem_range;
        shared_ptr<Metric> parent_metric_ptr;
        double parent_charge_time;

        Metric(int n, double t, double r, shared_ptr<Metric> p_n, double p_c) 
            : node_id(n), time_from_source(t), rem_range(r), parent_metric_ptr(p_n), parent_charge_time(p_c) {}

        Metric() 
            : node_id(-1), time_from_source(numeric_limits<double>::max()), rem_range(0.0), parent_metric_ptr(0), parent_charge_time(0.0) {}
};

bool is_better_metric(int node_id, list<Metric> & existing, Metric new_m) {
    for (auto ref_m : existing) {
        if (is_better_time(node_id, new_m.rem_range, new_m.time_from_source, ref_m.rem_range, ref_m.time_from_source)) {
            return false;
        }
    }
    // return existing.size() == (int)0;
    return true;
};

/*
Modified Breadth First Search (inspired from Bellman-Ford) algorithm with some imperfect heuristics to reduce the search space. 
Modifications: We will keep track of all possible <rem_range, time_from_source> combinations (Metric struct)
Heuristic: We have a notion of comparing two pairs of (reaching time and remaining range) at a station based
        on charge rate at that station, we use this heuristic to discard any un-promising pairs at any station to reduce the search space.
*/
void findMinTime_BfsLikeBellFord(int source, int target, vector<list<Edge> > & edges) {
    auto L = network.size();
    auto thisLevel = list<Metric>();
    auto none_metric = make_shared<Metric>();
    thisLevel.push_back(Metric(source, 0.0, FULL_RANGE, none_metric, 0.0));

    auto tracked_metrics = vector<list<Metric> >(L);

    for (int i = 0; i < L-1; ++i) {
        auto nextLevel = list<Metric>();
        for (auto m : thisLevel) {
            if (m.node_id == target) {
                continue;
            }
            auto m_ptr = make_shared<Metric>(m);
            for (auto e : edges[m.node_id]) {
                double drive_time = m.time_from_source + (e.cost / SPEED);
                // Case1: do full charge at m and reach the next edge
                double charge_time = (FULL_RANGE - m.rem_range) / network[m.node_id].rate;
                auto new_m = Metric(e.other_node, drive_time + charge_time, FULL_RANGE - e.cost, m_ptr, charge_time);
                if (e.other_node == target || is_better_metric(e.other_node, tracked_metrics[e.other_node], new_m)) {
                    nextLevel.push_back(new_m);
                    tracked_metrics[e.other_node].push_back(new_m);
                }
                // Case2: do minimal charge to reach destination
                Metric m;
                if (m.rem_range > e.cost) {
                    m = Metric(e.other_node, drive_time, m.rem_range - e.cost, m_ptr, 0.0);
                } else {
                    charge_time = (e.cost - m.rem_range) / network[m.node_id].rate;
                    m = Metric(e.other_node, drive_time + charge_time, 0.0, m_ptr, charge_time);
                }
                if (e.other_node == target || is_better_metric(e.other_node, tracked_metrics[e.other_node], m)) {
                    nextLevel.push_back(m);
                    tracked_metrics[e.other_node].push_back(m);
                }
            }
        }
        thisLevel = nextLevel;
    }

    // Find the best reaching time to target charger
    auto target_metrics = tracked_metrics[target];
    auto min = numeric_limits<double>::max();
    auto best = none_metric;
    for (auto m : target_metrics) {
        if (m.time_from_source < min) {
            min = m.time_from_source;
            best = make_shared<Metric>(m);
        }
    }

    // Some careful logic to print the charging stations and times along the best path.
    deque<int> charge_stations;
    deque<double> charge_times;
    while (best != none_metric) {
        charge_stations.push_front(best->node_id);
        if (best->parent_metric_ptr->node_id != source && best->node_id != source) {
            charge_times.push_front(best->parent_charge_time);
        }
        best = best->parent_metric_ptr;
    }
    auto it = charge_stations.begin();
    cout << network[*it].name;
    ++it;
    auto times_it = charge_times.begin();
    while (it != charge_stations.end()) {
        cout << ", " << network[*it].name;
        ++it;
        if (times_it != charge_times.end()) {
            cout << ", " << *times_it;
            ++times_it;
        }
    }
    cout << endl;
};


