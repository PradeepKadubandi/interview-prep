#include "network.h"
#include "math.h"
#include "Node.h"
#include "Edge.h"
#include<forward_list>
#include<unordered_map>
#include<list>
#include<queue>
using namespace std;
using namespace PK;

#define PI 3.14159265
#define EARTH_RADIUS 6356.752
#define FULL_RANGE 320.0
#define SPEED 105.0


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

class NodeMap {
    public:
        shared_ptr<Node> get_node_create_if_not_exists(row r) {
            auto key = r.name;
            if (_name_to_node_map.find(key) == _name_to_node_map.end()) {
                auto nodePtr = make_shared<Node>(r);
                _name_to_node_map.emplace(key, nodePtr);
            }
            return _name_to_node_map[key];
        }

        shared_ptr<Node> get_node_by_name(string name) {
            return _name_to_node_map[name];
        }

        void print() {
            cout << "-------------------------------------------------" << endl;
            cout << "               network" << endl;
            cout << "-------------------------------------------------" << endl;
            for (auto p : _name_to_node_map) {
                cout << p.first << " --> [ ";
                for (auto e : *(p.second->edges)) {
                    cout << e.other->data.name << ", " << e.cost << ";";
                }
                cout << endl;
            }
            cout << "-------------------------------------------------" << endl;
        }
    private:
        unordered_map<string, shared_ptr<Node> > _name_to_node_map;
};

shared_ptr<NodeMap> buildNodeMap() {
    shared_ptr<NodeMap> result = make_shared<NodeMap>();
    int edgeCount = 0;
    for (auto it1 = network.begin(); it1 < network.end(); ++it1) {
        for (auto it2 = it1+1; it2 < network.end(); ++it2) {
            auto n1 = result->get_node_create_if_not_exists(*it1);
            auto n2 = result->get_node_create_if_not_exists(*it2);
            auto distance = calculateGreatCircleDistance(*it1, *it2);
            if (distance <= FULL_RANGE) {
                // cout << "Distance Between " << (it1->name) << " and " << (it2->name) << " = " << distance << endl;
                n1->add_edge(Edge(n2, distance));
                n2->add_edge(Edge(n1, distance));
                edgeCount += 1;
            }
        }
    }
    // cout << "Total number of edges found = " << edgeCount << endl;
    return result;
};

struct IntermediateStation {
    public:
        string name;
        double charge_time;

        IntermediateStation(string n, double ct): name(n), charge_time(ct) {
        }
};

struct Result {
    public:
        double min_time;
        shared_ptr<list<IntermediateStation> > path;

        Result(double m_time, shared_ptr<list<IntermediateStation > > refPath) : min_time(m_time), path(refPath) {
        }
};

struct PrioirityQueueEntry {
    public:
        double min_time_to_reach_from_source;
        shared_ptr<Node> this_node_ptr;
        shared_ptr<Node> parent_node;

        PrioirityQueueEntry(double m, shared_ptr<Node> t_ptr, shared_ptr<Node> p_ptr) : min_time_to_reach_from_source(m), this_node_ptr(t_ptr), parent_node(p_ptr) {
        }

        bool operator >(const PrioirityQueueEntry& other) const{
            return min_time_to_reach_from_source > other.min_time_to_reach_from_source;
        }
};


shared_ptr<Result> findMinimumTimeGreedy(shared_ptr<Node> source, shared_ptr<Node> target) {
    priority_queue<PrioirityQueueEntry, vector<PrioirityQueueEntry>, greater<PrioirityQueueEntry> > pq;
    pq.emplace(PrioirityQueueEntry(0.0, source, 0));
    while (!pq.empty()) {
        
    }
    return make_shared<Result>(0.0, make_shared<list<IntermediateStation> >());
}

/*
Bruite force recrursive solution.
OPT(s, t) = Min [s -> v_i + OPT(v_i, t)] for all v_i in s.neighbors
For reaching s to v_i, we could consider 2 cases:
   just charge enough to reach v_i (rem charge at v_i will be zero)
   fully charge at s and then travel to v_i
visited : map of visited nodes along with the rem charge present when visting that node
path : stack of paths built as we visit nodes.

Returns the pair with minimum time to reach from s -> t as the first value,
the second value is a list of intermediate charging stations and times spent charging at those intermediate stations.
*/
shared_ptr<Result> findMinimumTimePath(shared_ptr<Node> start, shared_ptr<Node> destination, shared_ptr<unordered_map<string, double> > visited, shared_ptr<list<IntermediateStation > > path) {
    auto curr_charger = start->data;
    if (curr_charger.name == destination->data.name) {
        auto result_path = make_shared<list<IntermediateStation > >(*path);
        return make_shared<Result>(0.0, result_path);;
    }
    double min_value_so_far = numeric_limits<double>::max();
    shared_ptr<list<IntermediateStation > >result_path = NULL;
    for (auto edge : *start->edges) {
        auto remRange = visited->at(curr_charger.name);
        if (visited->find(edge.other->data.name) == visited->end()) {
            auto next_charger = edge.other->data;
            // Case1: charge fully here and go to other station
            // additional time = time to fully charge + time to reach destination, remRange at dest = FULL_RANGE - distance
            auto additional_time = edge.cost / SPEED;
            auto charge_time = (FULL_RANGE - remRange) / curr_charger.rate;
            additional_time += charge_time;
            auto next_rem_range = FULL_RANGE - edge.cost;
            visited->emplace(next_charger.name, next_rem_range);
            path->push_back(IntermediateStation(next_charger.name, charge_time));
            auto min_path_from_next = findMinimumTimePath(edge.other, destination, visited, path);
            visited->erase(next_charger.name);
            path->pop_back();
            auto total_time = additional_time + min_path_from_next->min_time;
            if (total_time < min_value_so_far) {
                // if (path->size() == 0) {
                //     cout << "New Min Path with cost " << total_time << endl;
                // }
                min_value_so_far = total_time;
                result_path = min_path_from_next->path;
            }
            // Case2: charge just the amount required to reach next station
            // additional time = min(0, distance - remRange) / rate + time to reach destination, remRange at dest = 0
            additional_time = edge.cost / SPEED;
            if (edge.cost > remRange) {
                charge_time = (edge.cost - remRange) / curr_charger.rate;
                additional_time += charge_time;
                next_rem_range = 0.0;
            }
            else {
                charge_time = 0.0;
                next_rem_range = remRange - edge.cost;
            }
            visited->emplace(next_charger.name, next_rem_range);
            path->push_back(IntermediateStation(next_charger.name, charge_time));
            min_path_from_next = findMinimumTimePath(edge.other, destination, visited, path);
            visited->erase(next_charger.name);
            path->pop_back();
            total_time = additional_time + min_path_from_next->min_time;
            if (total_time < min_value_so_far) {
                // if (path->size() == 0) {
                //     cout << "New Min Path with cost " << total_time << endl;
                // }
                min_value_so_far = total_time;
                result_path = min_path_from_next->path;
            }
        }
    }
    return make_shared<Result>(min_value_so_far, result_path);
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cout << "Error: requires initial and final supercharger names" << std::endl;        
        return -1;
    }
    
    std::string initial_charger_name = argv[1];
    std::string goal_charger_name = argv[2];

    auto nodeMap = buildNodeMap();
    // nodeMap->print();
    auto start = nodeMap->get_node_by_name(initial_charger_name);
    auto destination = nodeMap->get_node_by_name(goal_charger_name);
    auto visited = make_shared<unordered_map<string, double> >();
    visited->emplace(initial_charger_name, FULL_RANGE);
    auto path =  make_shared<list<IntermediateStation > >();
    auto min_time_and_path = findMinimumTimePath(start, destination, visited, path);

    string output = initial_charger_name;
    auto result_path = min_time_and_path->path;
    for (auto charger_time_pair : *result_path) {
        output += ", " + charger_time_pair.name + ", " + to_string(charger_time_pair.charge_time);
    }
    output += ", " + goal_charger_name;
    cout << output << endl;

    return 0;
}