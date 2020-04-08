#include <bits/stdc++.h>
using namespace std;

// Vertex count
const int V = 25000;
// Edge count
const int E = 280000;

// generate directed graph, without self-loop and multiple edge
int main(int argc, char* argv[])
{
    freopen("./test_data.txt", "w", stdout);

    // random number generater
    mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
    // generate integer in range [1, V] 
    uniform_int_distribution<int> distribution(1, V);

    map<pair<int, int>, bool> mp;

    int u, v;
    for(int i = 1; i <= E; ) {
        u = distribution(rng);
        v = distribution(rng);
        if(u == v || mp.count(make_pair(u,v)));
        else {
            i++;
            printf("%d,%d,1\n", u, v);
        }
    }

    return 0;
}
