#include <iostream>
#include <vector>
#include <random>
#include <tbb/task_group.h>
#include "Timer.h"

using namespace tbb;
using namespace std;

class node
{
public:
    node *right, *left;
    int data;

    node(node * r,node *l,int d)
    {
        right = r;
        left = l;
        data = d;
    }

    node(node &other)
    {
        right = other.right;
        left = other.left;
        data = other.data;
    }

};

node *root = nullptr;
//task_group g;

const int N = 1000000;
const int level = 5;

vector<int> input;
vector<int> ind(N);

int Fib(int n);

node * buildTree(vector<int>::iterator begin, vector<int>::iterator end, task_group &g)
{
    int num = distance(begin,end);

    if(num == 0) return nullptr;
    else if(num == 1)
    {
        node * temp = new node(nullptr,nullptr,input[*begin]);
        return temp;
    }
    else
    {
        int splitPos = num/2;
        std::nth_element(begin,begin+splitPos,end, [] (int i, int j) {return input[i] < input[j];});

        node *temp = new node(nullptr,nullptr,input[*(begin+splitPos)]);


        //task_group g;
        g.run([&]{temp->left = buildTree(begin,begin+splitPos,g);});
        g.run([&]{temp->right = buildTree(begin+splitPos+1,end,g);});
        g.wait();

        /*temp->left = buildTree(begin,begin+splitPos);
        temp->right = buildTree(begin+splitPos+1,end);*/
        return temp;
    }

}

void print(node * p)
{
    if(p!=nullptr)
    {
        print(p->left);
        cout<<p->data<<" ";
        print(p->right);
    }
}

int main()
{
    task_group g;

    std::fill(ind.begin(),ind.end(),0);
    for(int i=0;i<N;i++)
    {
        ind[i] = i;
    }

    std::random_device rd;
    std::uniform_int_distribution<int> distribution(1,N*10);

    for(int i=0;i<N;i++)
    {
        int k = distribution(rd);
        input.push_back(k);
    }

    /*
    cout<<"Input:";
    for(auto i=input.begin();i<input.end();i++)
    {
        cout<<*i<<" ";
    }*/

    knn::Timer timer;

    timer.Start();
    root = buildTree(ind.begin(),ind.end(),g);
    timer.Stop();
    cout<<"TIME FOR BUILD:"<<timer.Elapsed()<<endl;
    /*
    cout<<"\nIndex:";
    for(int i=0;i<N;i++)
    {
        cout<<ind[i]<<" ";
    }

    cout<<"\nTree(in-order):";
    print(root);
    cout<<endl;*/
    return 0;
}

int Fib(int n) {
    if( n<2 ) {
        return n;
    } else {
        long int x, y;
        task_group g;
        g.run([&]{x=Fib(n-1);}); // spawn a task
        g.run([&]{y=Fib(n-2);}); // spawn another task
        g.wait();                // wait for both tasks to complete
        return x+y;
    }
}
