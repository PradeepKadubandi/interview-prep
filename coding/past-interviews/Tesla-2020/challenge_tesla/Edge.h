#pragma once
#include<memory>
#include "Node.h"
using namespace std;

namespace PK {
    class Node;
    
    class Edge {
        public:
            shared_ptr<Node> other;
            double cost;

            Edge(shared_ptr<Node> other, double cost) : other(other), cost(cost) {
            }
    };
}