
cdef extern from '<Eigen/Eigen>' namespace 'Eigen':

    cdef cppclass Vector3d "Eigen::Matrix<double, 3, 1>":
        Matrix31d() except +
        double& operator[](size_t)

    cdef cppclass MatrixXd "Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>":
        MatrixXd() except +
        double * data()
        double& operator[](size_t)
        int rows()
        int cols()
        double coeff(int,int)
        double * resize(int,int)
