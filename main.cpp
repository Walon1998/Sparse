#include <chrono>
#include <functional>
#include <iostream>
#include <vector>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/SparseLU>

std::vector<Eigen::Triplet<double>> MakeTripletList(int n) {
    assert(n >= 2);
    int nnz = 3 * n - 2;
    std::vector<Eigen::Triplet<double>> tripletList(nnz);

    tripletList.emplace_back(Eigen::Triplet<double>(0, 0, 2));
    tripletList.emplace_back(Eigen::Triplet<double>(0, 1, -1));


    for (int i = 1, j = 0; i < n - 1; ++i, j++) {
        tripletList.emplace_back(Eigen::Triplet<double>(i, j, -1));
        tripletList.emplace_back(Eigen::Triplet<double>(i, j + 1, 2));
        tripletList.emplace_back(Eigen::Triplet<double>(i, j + 2, -1));

    }

    tripletList.emplace_back(Eigen::Triplet<double>(n - 1, n - 2, -1));
    tripletList.emplace_back(Eigen::Triplet<double>(n - 1, n - 1, 2));

    return tripletList;
}

double Runtime(const std::function<void(void)> &f) {

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();
    f();
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    return time_span.count();


}


template<class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
    os << "[";
    if (!v.empty()) {
        os << v[0];
        for (int i = 1; i < v.size(); ++i) os << ", " << v[i];
    }
    os << "]";

    return os;
}

int main() {
    // print small example of the tridiagonal matrix
    int m = 4;
    std::vector<Eigen::Triplet<double>> tripletList = MakeTripletList(m);
    Eigen::SparseMatrix<double> S_(m, m);
    S_.setFromTriplets(tripletList.begin(), tripletList.end());
    std::cout << "If n = " << m << ", then T equals" << std::endl;
    std::cout << Eigen::MatrixXd(S_) << std::endl;

    // matrix sizes for benchmark
    std::vector<int> N = {2, 4, 8, 16, 32, 64, 128, 256, 512};
    std::cout << "LU decomposition of T, where n = " << N << std::endl;

    // set up variables for runtime measurement
    std::vector<double> runtimeSparse;
    std::vector<double> runtimeDense;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> Sparesolver;
    Eigen::FullPivLU<Eigen::MatrixXd> Densesolver;

    for (int n : N) {
        tripletList = MakeTripletList(n);

        // sparse LU decomposition
        Eigen::SparseMatrix<double> S(n, n);
        S.setFromTriplets(tripletList.begin(), tripletList.end());






        // dense LU decomposition
        Eigen::MatrixXd D(S);

//        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
//        start = std::chrono::high_resolution_clock::now();
//        Sparesolver.compute(S);
//        end = std::chrono::high_resolution_clock::now();
//        std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(
//                end - start);
//
//        runtimeSparse.emplace_back(time_span.count());
//
//
//        start = std::chrono::high_resolution_clock::now();
//        Densesolver.compute(D);
//        end = std::chrono::high_resolution_clock::now();
//        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
//        runtimeDense.emplace_back(time_span.count());

       runtimeSparse.emplace_back(Runtime([S,&Sparesolver](){Sparesolver.compute(S);}));
        runtimeDense.emplace_back(Runtime([D,&Densesolver](){Densesolver.compute(D);}));
    }


    std::cout << "Runtime in seconds using storage format..." << std::endl;
    std::cout << "...sparse: " << runtimeSparse << std::endl;
    std::cout << "...dense:  " << runtimeDense << std::endl;

    return 0;
}
