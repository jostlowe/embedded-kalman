mod tests;

use crate::covariance_matrix::CovarianceMatrix;
use nalgebra::{Matrix2, RealField, SMatrix, SVector, Vector2};

trait Model<
    // numerical type use for computation. Typically f32 or f64
    T: RealField,
    // Size of the state vector x
    const Nx: usize,
    // Size of the measurement vector z
    const Nz: usize,
    // Size of the input vector u
    const Nu: usize,
>
{
    // State transition function
    fn f(x: SVector<T, Nx>, u: SVector<T, Nu>) -> SVector<T, Nx>;

    // Measurement function
    fn h(x: SVector<T, Nx>) -> SVector<T, Nz>;

    // State transition Jacobian
    fn f_jacobian(x: SVector<T, Nx>, u: SVector<T, Nu>) -> SMatrix<T, Nx, Nx>;

    // Measurement function Jacobain
    fn h_jacobian(x: SVector<T, Nx>) -> SMatrix<T, Nz, Nz>;
}

#[derive(Debug)]
pub struct KalmanFilter<T, const Nx: usize, const Nz: usize, const Nu: usize, S, M>
where
    T: RealField,
    S: State,
    M: Model<T, Nx, Nz, Nu>,
{
    model: M,
    state: S,
}

/*
impl<T, const Nx: usize, const Nz: usize, const Nu: usize>
KalmanFilter<T, Nx, Nz, Nu, Update<T, Nx, Nx>>
    where
        T: RealField,
{

    /// Gets the posterior state `x` and posterior state covariance `P` as a tuple.
    /// This method is only available when the Kalman Filter is in the `Update` state
    pub fn get_posteriors(&self) -> (&SVector<T, Nx>, &SMatrix<T, Nx, Nx>) {
        (&self.state.x_posterior, &self.state.P_posterior)
    }

    /// Use this method to perform the predict step of a Kalman Filter iteration,
    /// given an input vector `u`
    /// This method is only available when the Kalman Filter is in the `Update` state
    pub fn predict(self, u: SVector<T, Nu>) -> KalmanFilter<T, Nx, Nz, Nu, Predict<T, Nx, Nz>> {
        let Constants { F, B, Q, .. } = &self.constants;
        let Update {
            x_posterior,
            P_posterior,
            ..
        } = &self.state;
        let x_prior = F * x_posterior + B * u;
        let P_prior = F * P_posterior * F.transpose() + Q;

        KalmanFilter {
            constants: self.constants,
            state: Predict { P_prior, x_prior },
        }
    }

    /// Creates a new Kalman Filter. Look at the top-level documentation for usage and examples
    pub fn new(
        F: SMatrix<T, Nx, Nx>,
        B: SMatrix<T, Nx, Nu>,
        H: SMatrix<T, Nz, Nx>,
        Q: CovarianceMatrix<T, Nx>,
        R: CovarianceMatrix<T, Nz>,
        x0: SVector<T, Nx>,
        P0: CovarianceMatrix<T, Nx>,
    ) -> Self {
        Self {
            constants: Constants { F, B, H, Q: Q.0, R: R.0 },
            state: Update {
                x_posterior: x0,
                P_posterior: P0.0,
            },
        }
    }
}

impl<T, const Nx: usize, const Nz: usize, const Nu: usize>
KalmanFilter<T, Nx, Nz, Nu, Predict<T, Nx, Nx>>
    where
        T: RealField,
{
    /// Gets the prior state `x_prior` and prior state covariance `P_prior` as a tuple.
    /// This method is only available when the Kalman Filter is in the `Predict` state
    pub fn get_priors(&self) -> (&SVector<T, Nx>, &SMatrix<T, Nx, Nx>) {
        (&self.state.x_prior, &self.state.P_prior)
    }

    /// Use this method to perform the update step of a Kalman Filter iteration,
    /// given a measurement vector `z`. This method returns [None] when the innovation matrix
    /// can't be inverted.
    /// This method is only available when the Kalman Filter is in the `Update` state
    pub fn update(self, z: SVector<T, Nz>) -> Option<KalmanFilter<T, Nx, Nz, Nu, Update<T, Nx, Nz>>> {
        let Constants { H, R, .. } = &self.constants;
        let Predict {
            x_prior, P_prior, ..
        } = &self.state;

        // Innovation residual
        let y = z - H * x_prior;

        // Innovation covariance
        let S = H * P_prior * H.transpose() + R;

        // Kalman Gain
        let K = P_prior * H.transpose() * S.try_inverse()?;

        let x_posterior = x_prior + &K * y;
        let P_posterior = (SMatrix::identity() - &K * H) * P_prior;

        Some(KalmanFilter {
            constants: self.constants,
            state: Update {
                P_posterior,
                x_posterior,
            },
        })
    }
}


*/

#[derive(Debug)]
struct Constants<
    T: RealField,
    // Matrix Dimensions
    const Nx: usize,
    const Nz: usize,
    const Nu: usize,
    // Non-linear state function and measurement function
    F: Fn(SVector<T, Nx>, SVector<T, Nu>) -> SVector<T, Nx>,
    H: Fn(SVector<T, Nx>) -> SVector<T, Nz>,
    // State and measurement jacobians
    F_: Fn(SVector<T, Nx>, SVector<T, Nu>) -> SMatrix<T, Nx, Nx>,
    H_: Fn(SVector<T, Nx>) -> SMatrix<T, Nz, Nz>,
> {
    // Non-linear state function and measurement function
    f: F,
    h: H,

    // State and measurement jacobians
    F: F_,
    H: H_,

    // State transition noise matrix
    Q: SMatrix<T, Nx, Nx>,

    // Measurement noise matrix
    R: SMatrix<T, Nz, Nz>,
}

#[derive(Debug)]
struct Predict<T, const Nx: usize, const Nz: usize>
where
    T: RealField,
{
    // Prior state covariance matrix
    P_prior: SMatrix<T, Nx, Nx>,

    // State Vector
    x_prior: SVector<T, Nx>,
}

struct Update<T, const Nx: usize, const Nz: usize>
where
    T: RealField,
{
    // Prior state covariance matrix
    P_posterior: SMatrix<T, Nx, Nx>,

    // State Vector
    x_posterior: SVector<T, Nx>,
}

/// A marker trait to indicate what state the Kalman filter is currently in
pub trait State {}
impl<T, const Nx: usize, const Nz: usize> State for Predict<T, Nx, Nz> where T: RealField {}
impl<T, const Nx: usize, const Nz: usize> State for Update<T, Nx, Nz> where T: RealField {}

fn wolol() {
    let v = Vector2::new(1.0, 1.0);
    let a: Constants<f64, 2, 2, 2, _, _, _, _> = Constants {
        f: |x, u| x + v,
        h: |x| x,
        F: |x, u| Matrix2::new(1.0, 1.0, 1.0, 1.0),
        H: |x| Matrix2::new(1.0, 1.0, 1.0, 1.0),
        Q: Matrix2::new(1.0, 0.0, 0.0, 1.0),
        R: Matrix2::new(1.0, 0.0, 0.0, 1.0),
    };
}
