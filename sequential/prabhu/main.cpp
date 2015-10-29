#include <iostream>
#include <vector>
#include <queue>

using namespace std;

class point
{
    float x,y;
    int c;
    
public:
    
    point(float a, float b, int cl)
    {
        x=a;
        y=b;
        c=cl;
    }
    
    point()
    {
        
    }
    
    int getClassification()
    {
        return c;
    }
    
    float getx()
    {
        return x;
    }
    
    float gety()
    {
        return y;
    }
    
    float distance(point p)
    {
        return (this->x-p.getx())*(this->x-p.getx())+(this->y-p.gety())*(this->y-p.gety());
    }
    
    void getPoint()
    {
        cin>>this->x>>this->y>>this->c;
    }
    
};


class node
{
    
public:
    
    point p;
    float distance;
    
    node(point a, float b)
    {
        p = a;
        distance = b;
    }
};

struct comparator {
    bool operator()(node i, node j) {
        return i.distance > j.distance;
    }
};

int main()
{
    int k,N;
    cin>>k;
    cin>>N;
    
    vector <point> vector_p;
    point test;
    int i=0;
    
    while(N)
    {
        point temp;
        temp.getPoint();
        vector_p.push_back(temp);
        i++;
        N--;
    }
    
    test.getPoint();
    
    priority_queue <node, vector<node>, comparator >q;
    
    for(int i=0;i<vector_p.size();i++)
    {
        float d = vector_p[i].distance(test);
        node add(vector_p[i],d);
        q.push(add);
    }
    
    int c1,c2,c3;
    c1=c2=c3=0;
    
    for(int i=0;i<k;i++)
    {
        node temp = q.top();
        q.pop();
        if(temp.p.getClassification()==1) c1++;
        else if(temp.p.getClassification()==2) c2++;
        else c3++;
    }
    cout<<c1<<" "<<c2<<" "<<c3<<" ";
    cout<<(c1>c2?(c1>c3)?c1:c3:(c2>c3)?c2:c3);
    
    return 0;
}
