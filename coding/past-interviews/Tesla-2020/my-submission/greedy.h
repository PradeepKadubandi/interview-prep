#pragma once
#include "network.h"
#include "buildGraph.h"
#include "utilities.h"
#include<list>
#include<vector>
using namespace std;

#define FULL_RANGE 320.0
#define SPEED 105.0

/*
Incomplete implementation of a combination of greedy and BFS approach (inspired by Bellmon-Ford).
Greedy in the sense that only one combination of <reaching time, remaining range> is tracked per node.
This does not return a correct solution at all, so discarded the approach.
And then I wrote code in "bfs.h" that maintains a list of possible <reaching time, remaining range> pairs
for a given station. Since keeping tracking of all such possibilities makes the solution take a long time,
I chose a trade-off of discarding pairs that are not better than what we have seen so far.
*/
void findMinTimeGreedy(int source, int target, vector<list<Edge> > & edges) {
    auto L = network.size();
    auto inf = numeric_limits<double>::max();
    auto time_to_reach = vector<double>(L, inf);
    auto rem_range = vector<double>(L, 0.0);
    auto parents = vector<int>(L, -1);

    time_to_reach[source] = 0.0;
    rem_range[source] = FULL_RANGE;

    for (int i = 0; i < L-1; i++) {
        for (int node = 0; node < L; ++node) {
            if (time_to_reach[node] < inf) {
                for (auto e : edges[node]) {
                    if (parents[node] != e.other_node) {
                        double charge_time = 0.0;
                        double new_time_to_reach = time_to_reach[node] + (e.cost / SPEED);
                        double new_rem_range = 0.0;
                        if (rem_range[node] <= e.cost) {
                            charge_time = (e.cost - rem_range[node]) / network[node].rate;
                            new_time_to_reach += charge_time;
                        }
                        else {
                            new_rem_range = rem_range[node] - e.cost; 
                        }
                        if (is_better_time(e.other_node, rem_range[e.other_node], time_to_reach[e.other_node], new_rem_range, new_time_to_reach)) {
                            time_to_reach[e.other_node] = new_time_to_reach;
                            rem_range[e.other_node] = new_rem_range;
                            parents[e.other_node] = node;
                        }
                    }
                }
            }
        }
    }

    cout << "Time To Reach Target : " << time_to_reach[target] << endl;
    cout << "Remaining Range At Target : " << rem_range[target] << endl;
};
