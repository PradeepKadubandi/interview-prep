#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    void print(const vector<int> &nums) {
      for (auto i: nums) {
        cout << i << ", ";
      }
      cout << endl;
    }

    int firstMissingPositive(vector<int>& nums) {
        auto n = nums.size();
        for (auto& it: nums) {
            if (it < 0) {
                it = 0;
            }
        }
        cout << "After first iteration" << endl;
        print(nums);

        for (auto& it: nums) {
            if (it == 0) {
              continue;
            }
            auto idx = it;
            if (idx < 0) {
                idx *= -1;
            }
            if (idx <= n) {
                idx %= n;
                if (nums[idx] > 0) {
                    nums[idx] *= -1;
                }
                else if (nums[idx] == 0) {
                    nums[idx] = -1;
                }
            }
        }
        cout << "After second iteration" << endl;
        print(nums);

        for (auto i = 1; i < nums.size(); ++i) {
            if (nums[i] > 0) {
                return i;
            }
        }
        if (nums[0] > 0) {
          return n;
        }
        return n+1;
    }
};

int main() {
  Solution s;
  vector<int> ip({3, 4, -1, 1});
  auto r = s.firstMissingPositive(ip);
  cout << "Result : " << r << endl;
}