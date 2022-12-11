#include "network.h"
#include "utilities.h"
#include "buildGraph.h"
#include "bfs.h"
#include "astar.h"
#include<list>
#include<unordered_map>
#include<vector>
using namespace std;

#define FULL_RANGE 320.0
#define SPEED 105.0

void printEdges(vector<list<Edge> > edges) {
    for (int i = 0; i < edges.size(); i++) {
        cout << i << " --> ";
        for (auto e : edges[i]) {
            cout << e.other_node << ",";
        }
        cout << endl;
    }
};

void printNodeMap(unordered_map<string, int> node_name_to_node_id) {
    for (auto it : node_name_to_node_id) {
        cout << it.first << " : " << it.second << endl;
    }
};

void printDistances(vector<vector<double> > & distances) {
    for (int i = 0; i < distances.size(); ++i) {
        cout << i << " : ";
        for (int j = 0; j < distances.size(); ++j) {
            cout << "[" << j << "," << distances[i][j] << "]";
        }
        cout << endl;
    }
}

void test_extensive(vector<list<Edge> > & edges, vector<vector<double> > & distances) {
    for (int i = 0; i < network.size(); i++) {
        for (int j = 0; j < network.size(); j++) {
            if (i != j) {
                cout << i << "," << j << ":";
                findMinTimePath_AStar(i, j, edges, distances);
            }
        }
    }
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

    auto L = network.size();
    unordered_map<string, int> node_name_to_node_id;
    vector<list<Edge> > edges(L);
    vector<vector<double> > distances(L);
    buildGraph(node_name_to_node_id, edges, distances);

    // Just to visualize the graph, for debugging.
    // printNodeMap(node_name_to_node_id);
    // printEdges(edges);
    // printDistances(distances);

    // findMinTimeGreedy(node_name_to_node_id[initial_charger_name], node_name_to_node_id[goal_charger_name], edges);
    // findMinTime_BfsLikeBellFord(node_name_to_node_id[initial_charger_name], node_name_to_node_id[goal_charger_name], edges);
    findMinTimePath_AStar(node_name_to_node_id[initial_charger_name], node_name_to_node_id[goal_charger_name], edges, distances);

    // Run A-Star between all possible pairs.
    // test_extensive(edges, distances);

    // For debugging selected nodeId's easily.
    // findMinTimePath_AStar(stoi(initial_charger_name), stoi(goal_charger_name), edges, distances);
    return 0;
}