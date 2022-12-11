#pragma once
#include "network.h"
#include "utilities.h"
#include "buildGraph.h"
#include<deque>
#include<list>
#include<queue>
#include<unordered_set>
#include<vector>
using namespace std;

#define FULL_RANGE 320.0
#define SPEED 105.0

struct PQNode {
    public:
        double cost_to_come;
        double cost_to_go; // admissible heuristic cost to go
        double cost_heuristic; //cost-to-come + cost-to-go
        int node_id;
        double rem_range;
        shared_ptr<PQNode> parent_node_ptr;
        double parent_charge_time;

        PQNode(double c_t_c, double c_t_g, int n, double range, shared_ptr<PQNode> parent, double p_ct) : 
            cost_to_come(c_t_c), cost_to_go(c_t_g),
            cost_heuristic(c_t_c + c_t_g), node_id(n), rem_range(range),
            parent_node_ptr(parent), parent_charge_time(p_ct) {}

        bool operator >(const PQNode & other) const{
            // if (other.node_id != node_id) {
            //     return cost_heuristic > other.cost_heuristic;
            // }
            // return is_better_time(node_id, rem_range, cost_to_come, other.rem_range, other.cost_to_come);
            return cost_heuristic > other.cost_heuristic ||
                (cost_heuristic == other.cost_heuristic && rem_range > other.rem_range);
        }
};

void printPath(shared_ptr<PQNode> lastNode, int source) {
    deque<int> charge_stations;
    deque<double> charge_times;
    auto curr = lastNode;
    while (curr) {
        charge_stations.push_front(curr->node_id);
        if (curr->node_id != source && curr->parent_node_ptr->node_id != source) {
            charge_times.push_front(curr->parent_charge_time);
        }
        curr = curr->parent_node_ptr;
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
}

void findMinTimePath_AStar(int source, int target, const vector<list<Edge> > & edges, const vector<vector<double> > & distances) {
    auto pq = priority_queue<PQNode, vector<PQNode>, greater<PQNode> >();
    auto visited = unordered_set<int>();

    pq.push(PQNode(0.0, distances[source][target], source, FULL_RANGE, 0, 0.0));

    while (!pq.empty()) {
        auto c = pq.top();
        pq.pop();
        if (c.node_id == target) {
            printPath(make_shared<PQNode>(c), source);
            break;
        }
        if (visited.find(c.node_id) == visited.end()) {
            visited.insert(c.node_id);
            shared_ptr<PQNode> parent = make_shared<PQNode>(c);
            for (auto e: edges[c.node_id]) {
                auto drive_time = c.cost_to_come + (e.cost / SPEED);
                auto charge_time = 0.0;

                // Case1: charge fully at c and reach other node
                if (c.rem_range != FULL_RANGE) {
                    charge_time = (FULL_RANGE - c.rem_range) / network[c.node_id].rate;
                    pq.push(PQNode(drive_time + charge_time, distances[e.other_node][target], e.other_node, FULL_RANGE-e.cost, parent, charge_time));
                }

                // Case2: charge just required amount (zero if existing range is higher than distance) and reach other node
                charge_time = 0.0;
                auto new_rem_range = 0.0;
                if (c.rem_range >= e.cost) {
                    new_rem_range = c.rem_range - e.cost;
                } else {
                    // Adding a small value to distance is a quick hack to avoid infeasible paths due to close rounding off error.
                    charge_time = (e.cost - c.rem_range + 0.001) / network[c.node_id].rate;
                }
                pq.push(PQNode(drive_time + charge_time, distances[e.other_node][target], e.other_node, new_rem_range, parent, charge_time));
            }
        }
    }
}
