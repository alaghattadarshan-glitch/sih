#include "eskf_core.hpp"
#include <cmath>

namespace nav_core {

ESKFCore::ESKFCore(const ESKFConfig& cfg, double g_val)
    : config_(cfg), ins_(g_val), is_initialized_(false) {
    P_.setZero();
}

void ESKFCore::initialize(
    double initial_time,
    double lat_deg, double lon_deg, double alt_m,
    const Vec3& initial_vel_enu,
    const Quat& initial_quat,
    const Vec3& initial_accel_bias,
    const Vec3& initial_gyro_bias
) {
    ins_.initialize(
        initial_time, lat_deg, lon_deg, alt_m,
        initial_vel_enu, initial_quat,
        initial_accel_bias, initial_gyro_bias
    );

    P_.setZero();
    // Position covariance
    double p_var = config_.pos_std_init * config_.pos_std_init;
    P_.data[0][0] = p_var; P_.data[1][1] = p_var; P_.data[2][2] = p_var;

    // Velocity covariance
    double v_var = config_.vel_std_init * config_.vel_std_init;
    P_.data[3][3] = v_var; P_.data[4][4] = v_var; P_.data[5][5] = v_var;

    // Attitude covariance
    double att_var = config_.att_std_init * config_.att_std_init;
    P_.data[6][6] = att_var; P_.data[7][7] = att_var; P_.data[8][8] = att_var;

    // Accel bias covariance
    double ab_var = config_.accel_bias_std_init * config_.accel_bias_std_init;
    P_.data[9][9] = ab_var; P_.data[10][10] = ab_var; P_.data[11][11] = ab_var;

    // Gyro bias covariance
    double gb_var = config_.gyro_bias_std_init * config_.gyro_bias_std_init;
    P_.data[12][12] = gb_var; P_.data[13][13] = gb_var; P_.data[14][14] = gb_var;

    is_initialized_ = true;
}

void ESKFCore::buildErrorDynamics(
    const Mat3& R_b2n,
    const Vec3& f_body,
    double dt,
    Mat15& Phi,
    Mat15& Qd
) {
    // Continuous system matrix F (15x15)
    Mat15 F;
    F.setZero();

    // Specific force in ENU frame
    Vec3 f_enu = R_b2n * f_body;

    // dPos_dot = dVel
    F.data[0][3] = 1.0; F.data[1][4] = 1.0; F.data[2][5] = 1.0;

    // dVel_dot = -[f_enu x] * dAtt + R_b2n * dBa
    Mat3 f_skew = skew_symmetric(f_enu);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            F.data[3 + i][6 + j] = -f_skew.data[i][j];
            F.data[3 + i][9 + j] = R_b2n.data[i][j];
        }
    }

    // dAtt_dot = -R_b2n * dBg
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            F.data[6 + i][12 + j] = -R_b2n.data[i][j];
        }
    }

    // Discrete Transition Matrix Phi = I + F*dt + 0.5*(F*dt)^2
    Mat15 F_dt = F * dt;
    Mat15 F_dt_sq = F_dt * F_dt;
    Phi = Mat15::Identity() + F_dt + F_dt_sq * 0.5;

    // Continuous Noise Spectral Density Qc (12x12) -> G * Qc * G^T
    // G = [ 0   0   0   0 ] (pos: 3x12 zero)
    //     [ R   0   0   0 ] (vel: R * accel_noise)
    //     [ 0  -R   0   0 ] (att: -R * gyro_noise)
    //     [ 0   0   I   0 ] (ab : ab_rw)
    //     [ 0   0   0   I ] (gb : gb_rw)
    Mat15 GQcGt;
    GQcGt.setZero();

    double q_acc = config_.accel_noise_std * config_.accel_noise_std;
    double q_gyro = config_.gyro_noise_std * config_.gyro_noise_std;
    double q_ab_rw = config_.accel_bias_rw_std * config_.accel_bias_rw_std;
    double q_gb_rw = config_.gyro_bias_rw_std * config_.gyro_bias_rw_std;

    // G_vel * Qc_acc * G_vel^T = R * (q_acc * I) * R^T = q_acc * I (since R is orthogonal)
    GQcGt.data[3][3] = q_acc; GQcGt.data[4][4] = q_acc; GQcGt.data[5][5] = q_acc;

    // G_att * Qc_gyro * G_att^T = (-R) * (q_gyro * I) * (-R)^T = q_gyro * I
    GQcGt.data[6][6] = q_gyro; GQcGt.data[7][7] = q_gyro; GQcGt.data[8][8] = q_gyro;

    // Bias random walks
    GQcGt.data[9][9] = q_ab_rw; GQcGt.data[10][10] = q_ab_rw; GQcGt.data[11][11] = q_ab_rw;
    GQcGt.data[12][12] = q_gb_rw; GQcGt.data[13][13] = q_gb_rw; GQcGt.data[14][14] = q_gb_rw;

    // Qd = GQcGt * dt + 0.5 * (F * GQcGt + GQcGt * F^T) * dt^2
    Mat15 F_GQ = F * GQcGt;
    Mat15 GQ_FT = GQcGt * F.transpose();
    Qd = GQcGt * dt + (F_GQ + GQ_FT) * (0.5 * dt * dt);
    Qd.symmetrize();
}

NavState ESKFCore::predict(double timestamp, const Vec3& raw_accel, const Vec3& raw_gyro) {
    if (!is_initialized_) return ins_.getState();

    double t_prev = ins_.getState().timestamp;
    double dt = timestamp - t_prev;
    if (dt <= 0.0) return ins_.getState();

    NavState state_old = ins_.getState();
    Mat3 R_b2n = state_old.getRotationMatrix();
    Vec3 f_body = raw_accel - state_old.accel_bias;

    // 1. Nominal INS propagation
    NavState nom_state = ins_.update(timestamp, raw_accel, raw_gyro);

    // 2. Error covariance propagation
    Mat15 Phi, Qd;
    buildErrorDynamics(R_b2n, f_body, dt, Phi, Qd);

    P_ = Phi * P_ * Phi.transpose() + Qd;
    P_.symmetrize();

    return nom_state;
}

NavState ESKFCore::updateGNSS(const Vec3& gnss_pos_enu, double h_acc_m, double v_acc_m) {
    if (!is_initialized_) return ins_.getState();

    NavState nom = ins_.getState();
    Vec3 p_ins = nom.pos_enu;
    Vec3 innovation = gnss_pos_enu - p_ins;

    double h_std = (h_acc_m > 0.0) ? h_acc_m : config_.gnss_pos_std_default;
    double v_std = (v_acc_m > 0.0) ? v_acc_m : (h_std * 1.5);

    Mat3 R_gnss;
    R_gnss.data[0][0] = h_std * h_std;
    R_gnss.data[1][1] = h_std * h_std;
    R_gnss.data[2][2] = v_std * v_std;

    // S = P_pos + R_gnss
    Mat3 S;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            S.data[i][j] = P_.data[i][j] + R_gnss.data[i][j];
        }
    }

    Mat3 S_inv = S.inverse();

    // Kalman gain K = P * H^T * S^-1 (15x3)
    double K[15][3];
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += P_.data[i][k] * S_inv.data[k][j];
            }
            K[i][j] = sum;
        }
    }

    // 15-state error correction dx = K * innovation
    Vec15 dx;
    for (int i = 0; i < 15; ++i) {
        dx[i] = K[i][0] * innovation.x + K[i][1] * innovation.y + K[i][2] * innovation.z;
    }

    // Joseph-form covariance update: P = (I - K*H)*P*(I - K*H)^T + K*R*K^T
    Mat15 I_KH = Mat15::Identity();
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 3; ++j) {
            I_KH.data[i][j] -= K[i][j];
        }
    }

    Mat15 P_new = I_KH * P_ * I_KH.transpose();

    // Add K * R * K^T
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 15; ++j) {
            double krk = 0.0;
            for (int r = 0; r < 3; ++r) {
                krk += K[i][r] * R_gnss.data[r][r] * K[j][r];
            }
            P_new.data[i][j] += krk;
        }
    }
    P_ = P_new;
    P_.symmetrize();

    // Error Injection into nominal INS state
    nom.pos_enu += dx.getPos();
    nom.vel_enu += dx.getVel();

    Vec3 dtheta = dx.getAtt();
    Quat dq = Quat::fromRotationVector(dtheta);
    nom.q_b2n = Quat::multiply(nom.q_b2n, dq);

    nom.accel_bias += dx.getAccelBias();
    nom.gyro_bias += dx.getGyroBias();

    ins_.setState(nom);
    return nom;
}

ESKFCore::AIUpdateResult ESKFCore::updateAIDisplacement(
    const Vec3& ref_pos_enu,
    const Vec3& delta_p_ai,
    const Vec3& r_ai_std
) {
    AIUpdateResult result;
    result.state = ins_.getState();
    result.accepted = false;
    result.mahalanobis_dist = 0.0;

    if (!is_initialized_) return result;

    NavState nom = ins_.getState();
    Vec3 z_ai = ref_pos_enu + delta_p_ai;
    Vec3 innovation = z_ai - nom.pos_enu;

    Mat3 R_ai;
    R_ai.data[0][0] = r_ai_std.x * r_ai_std.x;
    R_ai.data[1][1] = r_ai_std.y * r_ai_std.y;
    R_ai.data[2][2] = r_ai_std.z * r_ai_std.z;

    Mat3 S;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            S.data[i][j] = P_.data[i][j] + R_ai.data[i][j];
        }
    }

    Mat3 S_inv = S.inverse();

    // Mahalanobis distance = sqrt(y^T * S^-1 * y)
    Vec3 Sinv_y = S_inv * innovation;
    double m_sq = innovation.dot(Sinv_y);
    double m_dist = std::sqrt(std::max(0.0, m_sq));
    result.mahalanobis_dist = m_dist;

    // Innovation gating check
    if (m_dist > config_.ai_gate_threshold) {
        return result; // Rejected
    }

    // Compute Kalman Gain K (15x3)
    double K[15][3];
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += P_.data[i][k] * S_inv.data[k][j];
            }
            K[i][j] = sum;
        }
    }

    // 15-state error correction dx = K * innovation
    Vec15 dx;
    for (int i = 0; i < 15; ++i) {
        dx[i] = K[i][0] * innovation.x + K[i][1] * innovation.y + K[i][2] * innovation.z;
    }

    // Joseph-form covariance update
    Mat15 I_KH = Mat15::Identity();
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 3; ++j) {
            I_KH.data[i][j] -= K[i][j];
        }
    }

    Mat15 P_new = I_KH * P_ * I_KH.transpose();
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 15; ++j) {
            double krk = 0.0;
            for (int r = 0; r < 3; ++r) {
                krk += K[i][r] * R_ai.data[r][r] * K[j][r];
            }
            P_new.data[i][j] += krk;
        }
    }
    P_ = P_new;
    P_.symmetrize();

    // Inject error into nominal state
    nom.pos_enu += dx.getPos();
    nom.vel_enu += dx.getVel();

    Vec3 dtheta = dx.getAtt();
    Quat dq = Quat::fromRotationVector(dtheta);
    nom.q_b2n = Quat::multiply(nom.q_b2n, dq);

    nom.accel_bias += dx.getAccelBias();
    nom.gyro_bias += dx.getGyroBias();

    ins_.setState(nom);
    result.state = nom;
    result.accepted = true;
    return result;
}

ESKFCore::NHCUpdateResult ESKFCore::updateNHC(
    double sigma_y,
    double sigma_z,
    double gate_threshold
) {
    NHCUpdateResult result;
    result.state = ins_.getState();
    result.accepted = false;
    result.mahalanobis_dist = 0.0;

    if (!is_initialized_) return result;

    NavState nom = ins_.getState();
    Mat3 R_b2n = nom.getRotationMatrix();
    Mat3 R_n2b = R_b2n.transpose();
    Vec3 v_body = R_n2b * nom.vel_enu;

    double vx_b = v_body.x;
    double vy_b = v_body.y;
    double vz_b = v_body.z;

    // Innovation y = z_nhc - h(x) = [0, 0]^T - [vy_b, vz_b]^T
    Vec2 innov(-vy_b, -vz_b);

    // Measurement noise covariance R_nhc (2x2)
    Mat2 R_nhc;
    R_nhc.data[0][0] = sigma_y * sigma_y;
    R_nhc.data[1][1] = sigma_z * sigma_z;

    // Measurement Jacobian H_nhc (2x15)
    double H[2][15];
    std::memset(H, 0, sizeof(H));

    // Row 0: Body Y velocity
    H[0][3] = R_b2n.data[0][1];
    H[0][4] = R_b2n.data[1][1];
    H[0][5] = R_b2n.data[2][1];
    H[0][6] = vz_b * R_b2n.data[0][0] - vx_b * R_b2n.data[0][2];
    H[0][7] = vz_b * R_b2n.data[1][0] - vx_b * R_b2n.data[1][2];
    H[0][8] = vz_b * R_b2n.data[2][0] - vx_b * R_b2n.data[2][2];

    // Row 1: Body Z velocity
    H[1][3] = R_b2n.data[0][2];
    H[1][4] = R_b2n.data[1][2];
    H[1][5] = R_b2n.data[2][2];
    H[1][6] = -vy_b * R_b2n.data[0][0] + vx_b * R_b2n.data[0][1];
    H[1][7] = -vy_b * R_b2n.data[1][0] + vx_b * R_b2n.data[1][1];
    H[1][8] = -vy_b * R_b2n.data[2][0] + vx_b * R_b2n.data[2][1];

    // Innovation covariance S = H * P * H^T + R_nhc (2x2)
    // First compute H_P (2x15) = H * P
    double H_P[2][15];
    for (int r = 0; r < 2; ++r) {
        for (int j = 0; j < 15; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 15; ++k) {
                sum += H[r][k] * P_.data[k][j];
            }
            H_P[r][j] = sum;
        }
    }

    Mat2 S;
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            double sum = 0.0;
            for (int k = 0; k < 15; ++k) {
                sum += H_P[r][k] * H[c][k];
            }
            S.data[r][c] = sum + R_nhc.data[r][c];
        }
    }

    // Gating check using 2x2 Mahalanobis distance
    double det_S = S.data[0][0] * S.data[1][1] - S.data[0][1] * S.data[1][0];
    if (std::abs(det_S) < 1e-12) {
        return result;
    }

    Mat2 S_inv;
    S_inv.data[0][0] = S.data[1][1] / det_S;
    S_inv.data[0][1] = -S.data[0][1] / det_S;
    S_inv.data[1][0] = -S.data[1][0] / det_S;
    S_inv.data[1][1] = S.data[0][0] / det_S;

    double mah_sq = innov.x * (S_inv.data[0][0] * innov.x + S_inv.data[0][1] * innov.y) +
                    innov.y * (S_inv.data[1][0] * innov.x + S_inv.data[1][1] * innov.y);
    double mah_dist = std::sqrt(std::max(0.0, mah_sq));
    result.mahalanobis_dist = mah_dist;

    if (mah_dist > gate_threshold) {
        return result;
    }

    // Kalman Gain K (15x2) = P * H^T * S^-1 = (H_P)^T * S^-1
    double K[15][2];
    for (int i = 0; i < 15; ++i) {
        double PHT_0 = 0.0;
        double PHT_1 = 0.0;
        for (int k = 0; k < 15; ++k) {
            PHT_0 += P_.data[i][k] * H[0][k];
            PHT_1 += P_.data[i][k] * H[1][k];
        }
        K[i][0] = PHT_0 * S_inv.data[0][0] + PHT_1 * S_inv.data[1][0];
        K[i][1] = PHT_0 * S_inv.data[0][1] + PHT_1 * S_inv.data[1][1];
    }

    // State error correction dx = K * innov (15x1)
    Vec15 dx;
    for (int i = 0; i < 15; ++i) {
        dx.data[i] = K[i][0] * innov.x + K[i][1] * innov.y;
    }

    // Joseph form covariance update
    Mat15 I_KH = Mat15::Identity();
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 15; ++j) {
            double kh = K[i][0] * H[0][j] + K[i][1] * H[1][j];
            I_KH.data[i][j] -= kh;
        }
    }

    Mat15 P_new;
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 15; ++j) {
            double sum_p = 0.0;
            for (int k = 0; k < 15; ++k) {
                for (int m = 0; m < 15; ++m) {
                    sum_p += I_KH.data[i][k] * P_.data[k][m] * I_KH.data[j][m];
                }
            }
            double krkt = K[i][0] * R_nhc.data[0][0] * K[j][0] +
                          K[i][0] * R_nhc.data[0][1] * K[j][1] +
                          K[i][1] * R_nhc.data[1][0] * K[j][0] +
                          K[i][1] * R_nhc.data[1][1] * K[j][1];
            P_new.data[i][j] = sum_p + krkt;
        }
    }
    P_ = P_new;
    P_.symmetrize();

    // Inject error into nominal state
    nom.pos_enu += dx.getPos();
    nom.vel_enu += dx.getVel();

    Vec3 dtheta = dx.getAtt();
    Quat dq = Quat::fromRotationVector(dtheta);
    nom.q_b2n = Quat::multiply(dq, nom.q_b2n);

    nom.accel_bias += dx.getAccelBias();
    nom.gyro_bias += dx.getGyroBias();

    ins_.setState(nom);
    result.state = nom;
    result.accepted = true;
    return result;
}

ESKFCore::ZUPTUpdateResult ESKFCore::updateZUPT(
    double sigma_zupt,
    double gate_threshold
) {
    ZUPTUpdateResult result;
    result.state = ins_.getState();
    result.accepted = false;
    result.mahalanobis_dist = 0.0;

    if (!is_initialized_) return result;

    NavState nom = ins_.getState();
    Vec3 innovation = -nom.vel_enu;

    Mat3 R_zupt;
    double var_z = sigma_zupt * sigma_zupt;
    R_zupt.data[0][0] = var_z;
    R_zupt.data[1][1] = var_z;
    R_zupt.data[2][2] = var_z;

    // S = P_vel + R_zupt (3x3)
    Mat3 S;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            S.data[i][j] = P_.data[3 + i][3 + j] + R_zupt.data[i][j];
        }
    }

    Mat3 S_inv = S.inverse();

    // Mahalanobis distance = sqrt(y^T * S^-1 * y)
    Vec3 Sinv_y = S_inv * innovation;
    double m_sq = innovation.dot(Sinv_y);
    double m_dist = std::sqrt(std::max(0.0, m_sq));
    result.mahalanobis_dist = m_dist;

    // Innovation gating check
    if (m_dist > gate_threshold) {
        return result; // Rejected by gating
    }

    // Kalman Gain K = P * H^T * S^-1 (15x3)
    // H has 1 at columns 3, 4, 5, so P * H^T is columns 3..5 of P
    double K[15][3];
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += P_.data[i][3 + k] * S_inv.data[k][j];
            }
            K[i][j] = sum;
        }
    }

    // 15-state error correction dx = K * innovation
    Vec15 dx;
    for (int i = 0; i < 15; ++i) {
        dx[i] = K[i][0] * innovation.x + K[i][1] * innovation.y + K[i][2] * innovation.z;
    }

    // Joseph form covariance update
    Mat15 I_KH = Mat15::Identity();
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 3; ++j) {
            I_KH.data[i][3 + j] -= K[i][j];
        }
    }

    Mat15 P_new = I_KH * P_ * I_KH.transpose();
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 15; ++j) {
            double krk = 0.0;
            for (int r = 0; r < 3; ++r) {
                krk += K[i][r] * R_zupt.data[r][r] * K[j][r];
            }
            P_new.data[i][j] += krk;
        }
    }
    P_ = P_new;
    P_.symmetrize();

    // Inject error into nominal state
    nom.pos_enu += dx.getPos();
    nom.vel_enu += dx.getVel();

    Vec3 dtheta = dx.getAtt();
    Quat dq = Quat::fromRotationVector(dtheta);
    nom.q_b2n = Quat::multiply(nom.q_b2n, dq);

    nom.accel_bias += dx.getAccelBias();
    nom.gyro_bias += dx.getGyroBias();

    ins_.setState(nom);
    result.state = nom;
    result.accepted = true;
    return result;
}

void ESKFCore::reset() {
    is_initialized_ = false;
    ins_.reset();
    P_.setZero();
}

} // namespace nav_core

