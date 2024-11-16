#include <bits/stdc++.h>
using namespace std;


#define ll long long
#define INF ((ll)(1e9+7))
#define fo(i, n) for(ll i=0;i<((ll)n);i++)
#define deb(x) cout << #x << " = " << (x) << endl;
#define deb2(x, y) cout << #x << " = " << (x) << ", " << #y << " = " << (y) << endl
#define pb push_back
#define mp make_pair
#define F first
#define S second
#define LSOne(S) ((S) & (-S))
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
typedef pair<ll, ll> pl;
typedef vector<ll> vl;
typedef vector<vl> vvl;
typedef vector<vpl> vvpl;
typedef vector<pl> vpl;
typedef vector<double> vd;
typedef vector<vd> vvd;
typedef vector<vvd> vvvd;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

vd sigmoid(vd x) {
    fo(i, x.size()) x[i] = 1.0 / (1.0 + exp(-x[i]));
    return x;
}

double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}


vd softmax(vd inp){
    double sum = 0;
    // double sum = 0.000001;
    fo(i, inp.size()){
        sum+=exp(inp[i]);
    }
    vd output;
    fo(i, inp.size()){
        output.pb(exp(inp[i])/sum);
    }
    return output;
}


struct NN{
    ll amLayers;
    vvd biases;
    vvvd weights; // lager, till, från
    vl layers;

    double lr = 0.01;

    NN(vl layers) : layers(layers){
        srand(1337);
        fo(i, layers.size()-1){
            biases.pb(vd(layers[i+1]));
            for(auto &b : biases[i]){
                b = ((double)rand()/(RAND_MAX))*0.01;
            }
        }
        fo(i, layers.size()-1){
            weights.pb({});
            fo(j, layers[i+1]){
                weights[i].pb({});
                fo(_, layers[i]){
                    weights[i][j].pb(((double)rand()/(RAND_MAX))*0.01);
                }
            }
        }
    }

    vd forward(vd currLayer){
        fo(i, amLayers-1){
            vd nextLayer;
            fo(j, layers[i+1]){
                double temp = biases[i][j];
                fo(k, layers[i]){
                    temp+=weights[i][j][k]*currLayer[k];
                }
                nextLayer.pb(sigmoid(temp));
            }
            swap(currLayer, nextLayer);
        }
        return softmax(currLayer);
    }

    void train(vd inLayer, vd correct){

        vvd z(amLayers-1), a(amLayers);
        a[0] = inLayer;
        fo(i, amLayers-1){
            z[i].resize(layers[i+1]);
            a[i+1].resize(layers[i+1]);
            
            fo(j, layers[i+1]){
                z[i][j] = biases[i][j];
                fo(k, layers[i]){
                    z[i][j] += weights[i][j][k] * a[i][k];
                }
            }
            a[i+1] = sigmoid(z[i]);
            if(i == amLayers-2){
                a[i+1] = softmax(z[i]);
            }
        }

        vvd delta(amLayers-1);
        
        delta[amLayers-2].resize(layers[amLayers-1]);
        fo(j, layers[amLayers-1]){
            delta[amLayers-2][j] = (a[amLayers-1][j] - correct[j]);
        }

        for(ll i = amLayers-3; i>=0; i--){
            delta[i].resize(layers[i+1]);
            fo(j, layers[i+1]){
                delta[i][j] = 0;
                fo(k, layers[i+2]){
                    delta[i][j] += delta[i+1][k] * weights[i+1][k][j];
                }
                delta[i][j] *= sigmoid_derivative(z[i][j]);
            }
        }

        fo(i, amLayers-1){
            fo(j, layers[i+1]){
                biases[i][j] -= lr * delta[i][j];
                fo(k, layers[i]){
                    weights[i][j][k] -= lr * delta[i][j] * a[i][k];
                }
            }
        }
    }
};



int main() {
    cin.tie(0)->sync_with_stdio(0);



    return 0;
}