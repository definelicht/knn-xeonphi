#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include "knn/KDTree.h"
#include "knn/Random.h"

using namespace std;
using namespace knn;

/* constexpr size_t n = 1<<1; */
constexpr size_t n = 4;
/* constexpr size_t nDims = 128; */
constexpr size_t nDims = 2;

using myTreeType = KDTree<nDims, false, float>;

void printTree(const myTreeType::NodeItr& a, const myTreeType::NodeItr& b)
{
    cout << *a.value() << endl;
    cout << *b.value() << endl;
    cout << endl;
    if (a.inBounds())
    {
        cout << "Left" << endl;
        printTree(a.Left(), b.Left());
    }
    if (a.inBounds())
    {
        cout << "Right" << endl;
        printTree(a.Right(), b.Right());
    }
}

void compareTree(const myTreeType::NodeItr& a, const myTreeType::NodeItr& b, const size_t level=0)
{
    cout << "Level = " << level << endl;
    cout << "Bounds a:   " << a.inBounds() << endl;
    cout << "Bounds b:   " << b.inBounds() << endl;
    cout << "Index a:    " << a.index() << endl;
    cout << "Index b:    " << b.index() << endl;
    cout << "SplitDim a: " << a.splitDim() << endl;
    cout << "SplitDim b: " << b.splitDim() << endl;
    cout << "Value a:    " << *a.value() << endl;
    cout << "Value b:    " << *b.value() << endl;
    cout << endl;
    assert(a.inBounds() == b.inBounds());
    assert(a.index() == b.index());
    assert(a.splitDim() == b.splitDim());
    assert(*a.value() == *b.value());
    if (a.inBounds())
        compareTree(a.Left(), b.Left(), level+1);
    if (a.inBounds())
        compareTree(a.Right(), b.Right(), level+1);
}


int main() {
    vector<float> train(n*nDims);
    Uniform(train.begin(), train.end());
    myTreeType kdTreeS(train, myTreeType::Pivot::median, 1, false);
    myTreeType kdTreeP(train, myTreeType::Pivot::median, 1, true);

    /* printTree(kdTreeS.Root(), kdTreeP.Root()); */
    compareTree(kdTreeS.Root(), kdTreeP.Root());

    return 0;
}
