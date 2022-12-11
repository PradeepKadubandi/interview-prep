#include <iostream>
#include <memory>
#include<unordered_map>
#include<list>
using namespace std;

int recursive_call(int start, int end, shared_ptr<unordered_map<string, int> > visited) {
  if (start == end) {
    return 0.0;
  }
  auto edges = list<int>();
  if (start > 0 && start <= 20) {
    for (int diff = -3; diff <= 3; ++diff) {
      edges.emplace_back(start + diff);
    }
  }
  int curValue = visited->at(to_string(start));
  int min_so_far = 100;
  for (auto edge: edges) {
    if (visited->find(to_string(edge)) == visited->end()) {
      visited->emplace(to_string(edge), curValue + edge - start);
      int recValue = recursive_call(edge, end, visited);
      if (curValue + recValue < min_so_far) {
        min_so_far = curValue + recValue;
      }
      visited->erase(to_string(edge));
    }
  }
  return min_so_far;
}

int main() {
  auto visited = make_shared<unordered_map<string, int> >();
  visited->emplace("1", 0);
  auto res = recursive_call(1, 5, visited);
  cout << res << endl;
}