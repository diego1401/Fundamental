// Imagine++ project
// Project:  Fundamental
// Author:   Pascal Monasse

#include "./Imagine/Features.h"
#include <Imagine/Graphics.h>
#include <Imagine/LinAlg.h>
#include <vector>
#include <cstdlib>
#include <ctime>
using namespace Imagine;
using namespace std;

static const float BETA = 0.01f; // Probability of failure

struct Match {
    float x1, y1, x2, y2;
};

// Display SIFT points and fill vector of point correspondences
void algoSIFT(Image<Color,2> I1, Image<Color,2> I2,
              vector<Match>& matches) {
    // Find interest points
    SIFTDetector D;
    D.setFirstOctave(-1);
    Array<SIFTDetector::Feature> feats1 = D.run(I1);
    drawFeatures(feats1, Coords<2>(0,0));
    cout << "Im1: " << feats1.size() << flush;
    Array<SIFTDetector::Feature> feats2 = D.run(I2);
    drawFeatures(feats2, Coords<2>(I1.width(),0));
    cout << " Im2: " << feats2.size() << flush;

    const double MAX_DISTANCE = 100.0*100.0;
    for(size_t i=0; i < feats1.size(); i++) {
        SIFTDetector::Feature f1=feats1[i];
        for(size_t j=0; j < feats2.size(); j++) {
            double d = squaredDist(f1.desc, feats2[j].desc);
            if(d < MAX_DISTANCE) {
                Match m;
                m.x1 = f1.pos.x();
                m.y1 = f1.pos.y();
                m.x2 = feats2[j].pos.x();
                m.y2 = feats2[j].pos.y();
                matches.push_back(m);
            }
        }
    }
}


// AUXILIARY FUNCTIONS
void fillA(vector<Match> matches, FMatrix<float,9,9>& A,FMatrix<float,3,3> N){
    float x1,x2,y1,y2;
    vector<int> indices(matches.size());
    iota(indices.begin(), indices.end(), 0);
    int index;
    random_shuffle(indices.begin(),indices.end());
    for(int i=0;i<9;i++){
        if(i==8){
            // Last equation, fill in 9-th row with 0s
            for(int j=0;j<8;j++) A(i,j) = 0;
        }
        else{
            index = indices[i];
            x1 = N(0,0) * matches[index].x1; y1 = N(1,1) * matches[index].y1;
            x2 = N(0,0) * matches[index].x2; y2 = N(1,1)  * matches[index].y2;

            A(i,0) = x1*x2; A(i,1) = y2*x1; A(i,2) = x1;
            A(i,3) = x2*y1; A(i,4) = y1*y2; A(i,5) = y1;
            A(i,6) = x2;    A(i,7) = y2;    A(i,8) = 1;
        }
    }
}

void enforeRanktwo(FMatrix<float,3,3>& F){
    FMatrix<float,3,3> U_f, V_transposed_f,Diag_S;
    FVector<float,3> S_f;
    Diag_S.fill(0);
    svd(F,U_f,S_f,V_transposed_f);
    S_f[2] = 0;
    for(int i=0; i < 3; i++) Diag_S(i,i) = S_f[i];
    F = U_f * Diag_S * V_transposed_f;
}

void solvingf(FMatrix<float,9,9> A ,FMatrix<float,3,3>& F){
    // variables for svd
    FVector<float,9> S;                      // Singular value decomposition:
    FMatrix<float,9,9> U, V_transposed;
    FVector<float,9> f;
    // Perform the SVD
    svd(A,U,S,V_transposed);
    // f is then the eigenvector associated to the smallest eigenvalue
    f = V_transposed.getRow(8);
    // Adding constraint
    for(int row=0;row<3;row++){
        for(int col=0;col<3;col++){
            F(row,col) = f[col + 3*row];
        }
    }
//    enforeRanktwo(F);
}

float model_classification(vector<Match> matches, FMatrix<float,3,3> F, vector<int>& currInliers, float distMax){
    int m = 0; // number of inliers
    float x1,x2,y1,y2,d;
    FVector<float,3> F_t_xi,xi,xi_prime;
    // Computing the distance
    for(int i=0;i<matches.size();i++){
        x1 = matches[i].x1; y1 = matches[i].y1;
        x2 = matches[i].x2; y2 = matches[i].y2;

        xi = {x1,y1,1};
        xi_prime = {x2,y2,1};
        F_t_xi = transpose(F) * xi;
        d = abs(F_t_xi[0]*xi_prime[0] + F_t_xi[1]*xi_prime[1] + F_t_xi[2]);
        d/= sqrt(F_t_xi[0]*F_t_xi[0] + F_t_xi[1]*F_t_xi[1]);
        if(d <= distMax){
            m ++;
            currInliers.push_back(i);
        }
    }
    return (float)m/ (float)matches.size();
    }

float update_Niter(int Niter, float curr_best_ratio,int k){
    int res = Niter;
    float tmp = ceil(log(BETA)/log(1-pow(curr_best_ratio,k)));
    if (tmp >= 0 and tmp < Niter) res = tmp + 1;
    return res;
}
// RANSAC algorithm to compute F from point matches (8-point algorithm)

void RANSAC(FMatrix<float,3,3>& bestF, vector<Match>& matches,vector<int>& bestInliers){
    vector<int> currInliers;
    const float distMax = 1.5f; // Pixel error for inlier/outlier discrimination
    float Niter=100000; // Adjusted dynamically
    float values_N[3][3]={{0.001,0,0},
                          {0,0.001,0},
                          {0,0,1}};
    FMatrix<float,3,3> N(values_N), F;
    FMatrix<float,9,9> A;
    int iteration_number = 0, k=8;
    float curr_ratio=0, curr_best_ratio = 0;;

    while(iteration_number < Niter){
        // Filling A to solve the linear system
        fillA(matches,A,N);
        // Solving and creating the matrix F
        solvingf(A,F);
        // Retrieve unnormalized F
        enforeRanktwo(F);
        F = transpose(N) * F * N;

        // Perform inlier/outlier classification
        curr_ratio = model_classification(matches,F,currInliers, distMax);

        if(curr_ratio > curr_best_ratio) {

            for(int row=0;row<3;row++){
                for(int col = 0;col<3;col++) bestF(row,col) = F(row,col);
            }
            //recompute Niter
            curr_best_ratio = curr_ratio;
            Niter = update_Niter(Niter,curr_best_ratio,k);

            bestInliers.clear();
            bestInliers = currInliers;
//            copy(currInliers.begin(),currInliers.end(),back_inserter(bestInliers));

        }

        iteration_number++;
        currInliers.clear();
    }

}

void leastSquares(FMatrix<float,3,3>& bestF,vector<Match> matches,FMatrix<float,3,3> N){
    Matrix<float> A(matches.size(),9);
    float x1,x2,y1,y2;
    // Important to normalize here too.
    for(int i=0;i<matches.size();i++){
            x1 = N(0,0) * matches[i].x1; y1 = N(1,1) * matches[i].y1;
            x2 = N(0,0) * matches[i].x2; y2 = N(1,1)  * matches[i].y2;

            A(i,0) = x1*x2; A(i,1) = y2*x1; A(i,2) = x1;
            A(i,3) = x2*y1; A(i,4) = y1*y2; A(i,5) = y1;
            A(i,6) = x2;    A(i,7) = y2;    A(i,8) = 1;
    }

    // variables for svd
    Vector<float> S;                      // Singular value decomposition:
    Matrix<float> U, V_transposed;
    Vector<float> f;
    // Perform the SVD
    svd(A,U,S,V_transposed);
    // f is then the eigenvector associated to the smallest eigenvalue
    f = V_transposed.getRow(8);
    // Adding constraint
    for(int row=0;row<3;row++){
        for(int col=0;col<3;col++){
            bestF(row,col) = f[col + 3*row];
        }
    }
    enforeRanktwo(bestF);
    bestF = transpose(N) * bestF * N;


}

// Parameter matches is filtered to keep only inliers as output.
FMatrix<float,3,3> computeF(vector<Match>& matches) {
    FMatrix<float,3,3> bestF;
    vector<int> bestInliers;
    int iteration_number = 0,k=8;
    float values_N[3][3]={{0.001,0,0},
                          {0,0.001,0},
                          {0,0,1}};
    FMatrix<float,3,3> N(values_N);

    RANSAC(bestF,matches,bestInliers);
    // Updating matches with inliers only
    vector<Match> all=matches;
    matches.clear();
    for(size_t i=0; i<bestInliers.size(); i++)
        matches.push_back(all[bestInliers[i]]);

    // Refining using only inliers
    leastSquares(bestF,matches,N);

    return bestF;
}

// Expects clicks in one image and show corresponding line in other image.
// Stop at right-click.
void displayEpipolar(Image<Color> I1, Image<Color> I2, const FMatrix<float,3,3>& F) {

    int x,y,start_x,end_x,start_y,end_y;
    FVector<float,3> u,v,point;
    bool clicked_image1;
    int counter = 0,counter_image1=0,counter_image2=0;
    Color c;
    while(true) {
        if(getMouse(x,y) == 3)
            break;

        //clicked I1 (left image)
        point[0] = (float)x; point[1] = (float)y; point[2] = 1;
        start_x = 0;

        clicked_image1 = x < I1.width();
        if(clicked_image1){
            //clicked on Image 1
            v = transpose(F) * point;
            end_x = I2.width();
        }
        else{
            //clicked Image 2 (right image)
            point[0] -=  I1.width(); // put in the same coordinate system as Image 2
            v = F * point;
            end_x = I1.width();

        }

        start_y = (int) (-(start_x*v[0] + v[2])/v[1]);
        end_y= (int) (-(end_x*v[0] + v[2])/v[1]);

        if (clicked_image1) {
            start_x = I1.width();
            end_x = I2.width()+I1.width();
            c = RED;
            counter_image1 ++;
            counter = counter_image1;
        }
        else{
            c = BLUE;
            counter_image2 ++;
            counter = counter_image2;
        }


        drawString((start_x+end_x)/2,(start_y+end_y)/2,to_string(counter),c);
        drawLine(start_x,start_y,
                 end_x,end_y,c);

        drawString(x,y,to_string(counter),c);
        drawCircle(x,y,1,c);

    }
}

int main(int argc, char* argv[])
{
    srand((unsigned int)time(0));

    const char* s1 = argc>1? argv[1]: srcPath("im1.jpg");
    const char* s2 = argc>2? argv[2]: srcPath("im2.jpg");

    // Load and display images
    Image<Color,2> I1, I2;
    if( ! load(I1, s1) ||
        ! load(I2, s2) ) {
        cerr<< "Unable to load images" << endl;
        return 1;
    }
    int w = I1.width();
    openWindow(2*w, I1.height());
    display(I1,0,0);
    display(I2,w,0);

    vector<Match> matches;
    algoSIFT(I1, I2, matches);
    const int n = (int)matches.size();
    cout << " matches: " << n << endl;
    drawString(100,20,std::to_string(n)+ " matches",RED);
    click();
    
    FMatrix<float,3,3> F = computeF(matches);
    cout << "F="<< endl << F;

    // Redisplay with matches
    display(I1,0,0);
    display(I2,w,0);
    for(size_t i=0; i<matches.size(); i++) {
        Color c(rand()%256,rand()%256,rand()%256);
        fillCircle(matches[i].x1+0, matches[i].y1, 2, c);
        fillCircle(matches[i].x2+w, matches[i].y2, 2, c);        
    }
    drawString(100, 20, to_string(matches.size())+"/"+to_string(n)+" inliers", RED);
    click();

    // Redisplay without SIFT points
    display(I1,0,0);
    display(I2,w,0);
    displayEpipolar(I1, I2, F);

    endGraphics();
    return 0;
}


