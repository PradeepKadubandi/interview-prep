#pragma once
#include<memory>
#include<forward_list>
#include "Edge.h"
#include "network.h"
using namespace std;

namespace PK {
    class Node {
        public:
            shared_ptr<forward_list<Edge> > edges;
            row data;

            Node(row data) : data(data) {
                edges = make_shared<forward_list<Edge> >();
            }

            bool operator ==(const Node& other) const{
                return data.name == other.data.name;
            }

            void add_edge(Edge edge) {
                edges->push_front(edge);
            }
    };
}