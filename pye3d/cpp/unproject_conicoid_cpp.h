#include <Eigen/Dense>

#include <cmath>
#include <exception>
#include <tuple>

// a = 0
template<typename T>
T solve(T a)
{
    if (a == 0) return 0;
    else throw std::runtime_error("No solution");
}
// ax + b = 0
template<typename T>
T solve(T a, T b)
{
    if (a == 0) return solve(b);

    return -b / a;
}
// ax^2 + bx + c = 0
template<typename T>
std::tuple<T, T> solve(T a, T b, T c)
{
    using std::sqrt;

    if (a == 0) {
        auto root = solve(b, c);
        return std::tuple<T, T>(root, root);
    }

    // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-6.pdf
    // Pg 184
    auto det = (b * b) - 4 * a * c;

    if (det < 0)
        throw std::runtime_error("No solution");

    //auto sqrtdet = sqrt(det);
    auto q = -0.5 * (b + (b >= 0 ? 1 : -1) * sqrt(det));
    return std::tuple<T, T>(q / a, c / q);
}
// ax^2 + bx + c = 0
template<typename T>
std::tuple<T, T, T> solve(T a, T b, T c, T d)
{
    using std::sqrt;
    using std::abs;
    using std::cbrt;

    if (a == 0) {
        auto roots = solve(b, c, d);
        return std::tuple<T, T, T>(std::get<0>(roots), std::get<1>(roots), std::get<1>(roots));
    }

    // http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/c5-6.pdf
    // http://web.archive.org/web/20120321013251/http://linus.it.uts.edu.au/~don/pubs/solving.html
    auto p = b / a;
    auto q = c / a;
    auto r = d / a;
    //auto Q = (p*p - 3*q) / 9;
    //auto R = (2*p*p*p - 9*p*q + 27*r)/54;
    auto u = q - (p * p) / 3;
    auto v = r - p * q / 3 + 2 * p * p * p / 27;
    auto j = 4 * u * u * u / 27 + v * v;
    const auto M = std::numeric_limits<T>::max();
    const auto sqrtM = sqrt(M);
    const auto cbrtM = cbrt(M);

    if (b == 0 && c == 0)
        return std::tuple<T, T, T>(cbrt(-d), cbrt(-d), cbrt(-d));

    if (abs(p) > 27 * cbrtM)
        return std::tuple<T, T, T>(-p, -p, -p);

    if (abs(q) > sqrtM)
        return std::tuple<T, T, T>(-cbrt(v), -cbrt(v), -cbrt(v));

    if (abs(u) > 3 * cbrtM / 4)
        return std::tuple<T, T, T>(cbrt(4) * u / 3, cbrt(4) * u / 3, cbrt(4) * u / 3);

    if (j > 0) {
        // One real root
        auto w = sqrt(j);
        T y;

        if (v > 0)
            y = (u / 3) * cbrt(2 / (w + v)) - cbrt((w + v) / 2) - p / 3;
        else
            y = cbrt((w - v) / 2) - (u / 3) * cbrt(2 / (w - v)) - p / 3;

        return std::tuple<T, T, T>(y, y, y);

    } else {
        // Three real roots
        auto s = sqrt(-u / 3);
        auto t = -v / (2 * s * s * s);
        auto k = acos(t) / 3;
        auto y1 = 2 * s * cos(k) - p / 3;
        auto y2 = s * (-cos(k) + sqrt(3.) * sin(k)) - p / 3;
        auto y3 = s * (-cos(k) - sqrt(3.) * sin(k)) - p / 3;
        return std::tuple<T, T, T>(y1, y2, y3);
    }
}

struct Circle3D
{
    Eigen::Vector3d center;
    Eigen::Vector3d normal;
    double radius;
};

template <typename T>
inline int sign(T val)
{
    // type-safe signum function, returns -1, 0 or 1
    return (T(0) < val) - (val < T(0));
}

std::pair<Circle3D, Circle3D> unproject_conicoid(
    const double a,
    const double b,
    const double c,
    const double f,
    const double g,
    const double h,
    const double u,
    const double v,
    const double w,
    const double focal_length,
    const double circle_radius
)
{
    using std::sqrt;
    using std::abs;
    
    typedef Eigen::Matrix<double, 3, 3> Matrix3;
    typedef Eigen::Matrix<double, 3, 1> Vector3;
    typedef Eigen::Array<double, 1, 3> RowArray3;
    typedef Eigen::Translation<double, 3> Translation3;

    //auto d = pupil_cone.D;
    // Get canonical conic form:
    //     lambda(1) X^2 + lambda(2) Y^2 + lambda(3) Z^2 = mu
    // Safaee-Rad 1992 eq (6)
    // Done by solving the discriminating cubic (10)
    // Lambdas are sorted descending because order of roots doesn't
    // matter, and it later eliminates the case of eq (30), where
    // lambda(2) > lambda(1)
    RowArray3 lambda;
    std::tie(lambda(0), lambda(1), lambda(2)) = solve(1., -(a + b + c), (b * c + c * a + a * b - f * f - g * g - h * h), -(a * b * c + 2 * f * g * h - a * f * f - b * g * g - c * h * h));
    assert(lambda(0) >= lambda(1));
    assert(lambda(1) > 0);
    assert(lambda(2) < 0);
    // Now want to calculate l,m,n of the plane
    //     lX + mY + nZ = p
    // which intersects the cone to create a circle.
    // Safaee-Rad 1992 eq (31)
    // [Safaee-Rad 1992 eq (33) comes out of this as a result of lambda(1) == lambda(2)]
    auto n = sqrt((lambda(1) - lambda(2)) / (lambda(0) - lambda(2)));
    auto m = 0.0;
    auto l = sqrt((lambda(0) - lambda(1)) / (lambda(0) - lambda(2)));
    // There are two solutions for l, positive and negative, we handle these later
    // Want to calculate T1, the rotation transformation from image
    // space in the canonical conic frame back to image space in the
    // real world
    Matrix3 T1;
    // Safaee-Rad 1992 eq (8)
    auto li = T1.row(0);
    auto mi = T1.row(1);
    auto ni = T1.row(2);
    // Safaee-Rad 1992 eq (12)
    RowArray3 t1 = (b - lambda) * g - f * h;
    RowArray3 t2 = (a - lambda) * f - g * h;
    RowArray3 t3 = -(a - lambda) * (t1 / t2) / g - h / g;
    mi = 1 / sqrt(1 + (t1 / t2).square() + t3.square());
    li = (t1 / t2) * mi.array();
    ni = t3 * mi.array();

    // If li,mi,ni follow the left hand rule, flip their signs
    if ((li.cross(mi)).dot(ni) < 0) {
        li = -li;
        mi = -mi;
        ni = -ni;
    }

    // Calculate T2, a translation transformation from the canonical
    // conic frame to the image space in the canonical conic frame
    // Safaee-Rad 1992 eq (14)
    Translation3 T2;
    T2.translation() = -(u * li + v * mi + w * ni).array() / lambda;
    Circle3D solutions[2];
    double ls[2] = { l, -l };

    for (int i = 0; i < 2; i++) {
        auto l = ls[i];
        // Circle normal in image space (i.e. gaze vector)
        Vector3 gaze = T1 * Vector3(l, m, n);
        // Calculate T3, a rotation from a frame where Z is the circle normal
        // to the canonical conic frame
        // Safaee-Rad 1992 eq (19)
        // Want T3 = / -m/sqrt(l*l+m*m) -l*n/sqrt(l*l+m*m) l \
        //              |  l/sqrt(l*l+m*m) -m*n/sqrt(l*l+m*m) m |
        //                \            0           sqrt(l*l+m*m)   n /
        // But m = 0, so this simplifies to
        //      T3 = /       0      -n*l/sqrt(l*l) l \
        //              |  l/sqrt(l*l)        0       0 |
        //                \          0         sqrt(l*l)   n /
        //         = /    0    -n*sgn(l) l \
        //              |  sgn(l)     0     0 |
        //                \       0       |l|    n /
        Matrix3 T3;

        if (l == 0) {
            // Discontinuity of sgn(l), have to handle explicitly
            assert(n == 1);
            T3 << 0, -1, 0,
            1, 0, 0,
            0, 0, 1;

        } else {
            //auto sgnl = sign(l);
            T3 << 0, -n* sign(l), l,
            sign(l), 0, 0,
            0, abs(l), n;
        }

        // Calculate the circle center
        // Safaee-Rad 1992 eq (38), using T3 as defined in (36)
        auto A = lambda.matrix().dot(T3.col(0).cwiseAbs2());
        auto B = lambda.matrix().dot(T3.col(0).cwiseProduct(T3.col(2)));
        auto C = lambda.matrix().dot(T3.col(1).cwiseProduct(T3.col(2)));
        auto D = lambda.matrix().dot(T3.col(2).cwiseAbs2());
        // Safaee-Rad 1992 eq (41)
        Vector3 center_in_Xprime;
        center_in_Xprime(2) = A * circle_radius / sqrt((B * B) + (C * C) - A * D);
        center_in_Xprime(0) = -B / A * center_in_Xprime(2);
        center_in_Xprime(1) = -C / A * center_in_Xprime(2);
        // Safaee-Rad 1992 eq (34)
        Translation3 T0;
        T0.translation() << 0, 0, focal_length;
        // Safaee-Rad 1992 eq (42) using (35)
        Vector3 center = T0 * T1 * T2 * T3 * center_in_Xprime;

        // If z is negative (behind the camera), choose the other
        // solution of eq (41) [maybe there's a way of calculating which
        // solution should be chosen first]

        if (center(2) < 0) {
            center_in_Xprime = -center_in_Xprime;
            center = T0 * T1 * T2 * T3 * center_in_Xprime;
        }

        // Make sure that the gaze vector is toward the camera and is normalised
        if (gaze.dot(center) > 0) {
            gaze = -gaze;
        }

        gaze.normalize();
        // Save the results
        Circle3D& circle = solutions[i];
        circle.center = center;
        circle.normal = gaze;
        circle.radius = circle_radius;
    }

    return std::make_pair(solutions[0], solutions[1]);
}
