#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Vertex count
const int V = 25000;
// Edge count
const int E = 280000;

// generate directed graph, without self-loop and multiple edge
int main(int argc, char* argv[])
{
    registerGen(argc, argv, 1);
    freopen("./test_data.txt", "w", stdout);

    map<pair<int, int>, bool> mp;

    int u, v;
    for(int i = 1; i <= E; ) {
        u = rnd.next(1, V);
        v = rnd.next(1, V);
        if(u == v || mp.count(make_pair(u,v)));
        else {
            i++;
            printf("%d,%d,1\n", u, v);
        }
    }

    return 0;
}
