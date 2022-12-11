#pragma once
#include "network.h"
#include "utilities.h"
#include<unordered_map>
#include<list>
#include<vector> 
using namespace std;

#define FULL_RANGE 320.0

class Edge {
    public:
        int other_node;
        double cost;

        Edge(int o, double c) : other_node(o), cost(c) {}
};

/*
In this method, 'edges' is built with only 'feasible' edges i.e., those with distance < FULL_RANGE while distances is 2-D array of all possible distances.
Having both of them is redundant, we can merge them both and change the edge traversal in graph to filter based on FULL_RANGE.
The code is like this more for historical reason, in the first approaches that I tried (before A*),
I didn't need pair wise distances, so only 'feasible' edges are built as part of this method.
When I implemented A*, I needed heuristic distances between every pair (strictly speaking only to the goal are needed for a single run)
and I added 'distances' 2-D vector to avoid recalculating great circle distance to goal every time. 
I could easily fix the code but thought that for this exercise, it's not a big deal.
*/
void buildGraph(unordered_map<string, int> & node_name_to_node_id, vector<list<Edge> > & edges, vector<vector<double> > & distances) {
    auto L = network.size();
    for (int i = 0; i < L; ++i) {
        distances[i] = vector<double>(L);
    }
    for (int i = 0; i < L; ++i) {
        node_name_to_node_id.emplace(network[i].name, i);
        for (int j = i+1; j < L; j++) {
            auto distance = calculateGreatCircleDistance(network[i], network[j]);
            if (distance <= FULL_RANGE) {
                edges[i].push_back(Edge(j, distance));
                edges[j].push_back(Edge(i, distance));
            }
            distances[i][j] = distance;
            distances[j][i] = distance;
        }
    }
};