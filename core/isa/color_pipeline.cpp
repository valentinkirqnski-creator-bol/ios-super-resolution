#include "color_pipeline.h"

#include <cmath>
#include <algorithm>

namespace isacpu {

namespace {

struct Vec3d { double x, y, z; };
struct XY { double x, y; };

double pin(double lo, double v, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

Vec3d mat3_apply(const Mat3d& M, const Vec3d& v) {
    return {M.m[0] * v.x + M.m[1] * v.y + M.m[2] * v.z,
            M.m[3] * v.x + M.m[4] * v.y + M.m[5] * v.z,
            M.m[6] * v.x + M.m[7] * v.y + M.m[8] * v.z};
}

Mat3d mat3_mul(const Mat3d& A, const Mat3d& B) {
    Mat3d C{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) {
            double sum = 0;
            for (int k = 0; k < 3; ++k) sum += A.m[r * 3 + k] * B.m[k * 3 + c];
            C.m[r * 3 + c] = sum;
        }
    return C;
}

Mat3d mat3_add(const Mat3d& A, const Mat3d& B, double wa, double wb) {
    Mat3d C{};
    for (int i = 0; i < 9; ++i) C.m[i] = A.m[i] * wa + B.m[i] * wb;
    return C;
}

Mat3d mat3_scale(const Mat3d& A, double s) {
    Mat3d C{};
    for (int i = 0; i < 9; ++i) C.m[i] = A.m[i] * s;
    return C;
}

Mat3d mat3_diag(double a, double b, double c) { return {a, 0, 0, 0, b, 0, 0, 0, c}; }

Mat3d mat3_invert(const Mat3d& M) {
    double a = M.m[0], b = M.m[1], c = M.m[2];
    double d = M.m[3], e = M.m[4], f = M.m[5];
    double g = M.m[6], h = M.m[7], i = M.m[8];

    double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    double invdet = det != 0.0 ? 1.0 / det : 0.0;

    Mat3d R;
    R.m[0] = (e * i - f * h) * invdet;
    R.m[1] = (c * h - b * i) * invdet;
    R.m[2] = (b * f - c * e) * invdet;
    R.m[3] = (f * g - d * i) * invdet;
    R.m[4] = (a * i - c * g) * invdet;
    R.m[5] = (c * d - a * f) * invdet;
    R.m[6] = (d * h - e * g) * invdet;
    R.m[7] = (b * g - a * h) * invdet;
    R.m[8] = (a * e - b * d) * invdet;
    return R;
}

Vec3d xy_to_xyz(XY xy) {
    double x = pin(0.000001, xy.x, 0.999999);
    double y = pin(0.000001, xy.y, 0.999999);
    if (x + y > 0.999999) {
        double scale = 0.999999 / (x + y);
        x *= scale;
        y *= scale;
    }
    return {x / y, 1.0, (1.0 - x - y) / y};
}

const XY kD50{0.3457, 0.3585};

XY xyz_to_xy(Vec3d v) {
    double total = v.x + v.y + v.z;
    if (total > 0.0) return {v.x / total, v.y / total};
    return kD50;
}

const XY kPCStoXY = kD50;
const Vec3d kPCStoXYZ = xy_to_xyz(kD50);

const Mat3d kBradford{0.8951, 0.2664, -0.1614,
                      -0.7502, 1.7135, 0.0367,
                       0.0389, -0.0685, 1.0296};

Mat3d map_white_matrix(XY white1, XY white2) {
    Vec3d w1 = mat3_apply(kBradford, xy_to_xyz(white1));
    Vec3d w2 = mat3_apply(kBradford, xy_to_xyz(white2));
    w1 = {std::max(w1.x, 0.0), std::max(w1.y, 0.0), std::max(w1.z, 0.0)};
    w2 = {std::max(w2.x, 0.0), std::max(w2.y, 0.0), std::max(w2.z, 0.0)};

    Mat3d A = mat3_diag(pin(0.1, w1.x > 0.0 ? w2.x / w1.x : 10.0, 10.0),
                        pin(0.1, w1.y > 0.0 ? w2.y / w1.y : 10.0, 10.0),
                        pin(0.1, w1.z > 0.0 ? w2.z / w1.z : 10.0, 10.0));

    Mat3d bradfordInv = mat3_invert(kBradford);
    return mat3_mul(mat3_mul(bradfordInv, A), kBradford);
}

// EXIF/DNG LightSource codes -> correlated colour temperature (Kelvin),
// matching DNGColorSpec.ConvertIlluminantToTemperature exactly.
double illuminant_to_temperature(int illuminant) {
    switch (illuminant) {
        case 1: return 5500.0;                    // Daylight
        case 2: return (3800.0 + 4500.0) * 0.5;    // Fluorescent
        case 3: return 2850.0;                     // Tungsten
        case 4: return 5500.0;                     // Flash
        case 9: return 5500.0;                     // FineWeather
        case 10: return 6500.0;                    // CloudyWeather
        case 11: return 7500.0;                    // Shade
        case 12: return (5700.0 + 7100.0) * 0.5;   // DaylightFluorescent
        case 13: return (4600.0 + 5500.0) * 0.5;   // DayWhiteFluorescent
        case 14: return (3800.0 + 4500.0) * 0.5;   // CoolWhiteFluorescent
        case 15: return (3250.0 + 3800.0) * 0.5;   // WhiteFluorescent
        case 16: return (2600.0 + 3250.0) * 0.5;   // WarmWhiteFluorescent
        case 17: return 2850.0;                    // StandardLightA
        case 18: return 5500.0;                    // StandardLightB
        case 19: return 6500.0;                    // StandardLightC
        case 20: return 5500.0;                    // D55
        case 21: return 6500.0;                    // D65
        case 22: return 7500.0;                    // D75
        case 23: return 5000.0;                    // D50
        case 24: return 3200.0;                     // ISOStudioTungsten
        default: return 0.0;                       // Unknown / OtherLightSource
    }
}

struct Ruvt { double r, u, v, t; };
const Ruvt kTempTable[31] = {
    {0, 0.18006, 0.26352, -0.24341}, {10, 0.18066, 0.26589, -0.25479}, {20, 0.18133, 0.26846, -0.26876},
    {30, 0.18208, 0.27119, -0.28539}, {40, 0.18293, 0.27407, -0.30470}, {50, 0.18388, 0.27709, -0.32675},
    {60, 0.18494, 0.28021, -0.35156}, {70, 0.18611, 0.28342, -0.37915}, {80, 0.18740, 0.28668, -0.40955},
    {90, 0.18880, 0.28997, -0.44278}, {100, 0.19032, 0.29326, -0.47888}, {125, 0.19462, 0.30141, -0.58204},
    {150, 0.19962, 0.30921, -0.70471}, {175, 0.20525, 0.31647, -0.84901}, {200, 0.21142, 0.32312, -1.0182},
    {225, 0.21807, 0.32909, -1.2168}, {250, 0.22511, 0.33439, -1.4512}, {275, 0.23247, 0.33904, -1.7298},
    {300, 0.24010, 0.34308, -2.0637}, {325, 0.24702, 0.34655, -2.4681}, {350, 0.25591, 0.34951, -2.9641},
    {375, 0.26400, 0.35200, -3.5814}, {400, 0.27218, 0.35407, -4.3633}, {425, 0.28039, 0.35577, -5.3762},
    {450, 0.28863, 0.35714, -6.7262}, {475, 0.29685, 0.35823, -8.5955}, {500, 0.30505, 0.35907, -11.324},
    {525, 0.31320, 0.35968, -15.628}, {550, 0.32129, 0.36011, -23.325}, {575, 0.32931, 0.36038, -40.770},
    {600, 0.33724, 0.36051, -116.45},
};

// DNGTemperature's xyCoord SETTER (xy -> temperature), Adobe's Robertson-
// table method. Only the temperature is needed downstream (tint is
// computed but unused by FindXYZtoCamera).
double temperature_from_xy(XY xy) {
    double u = 2.0 * xy.x / (1.5 - xy.x + 6.0 * xy.y);
    double v = 3.0 * xy.y / (1.5 - xy.x + 6.0 * xy.y);

    double last_dt = 0.0, last_du = 0.0, last_dv = 0.0;
    double temperature = 5000.0;

    for (int index = 1; index <= 30; ++index) {
        double du = 1.0, dv = kTempTable[index].t;
        double len = std::sqrt(1.0 + dv * dv);
        du /= len; dv /= len;

        double uu = u - kTempTable[index].u;
        double vv = v - kTempTable[index].v;
        double dt = -uu * dv + vv * du;

        if (dt <= 0.0 || index == 30) {
            if (dt > 0.0) dt = 0.0;
            dt = -dt;

            double f = (index == 1) ? 0.0 : dt / (last_dt + dt);
            temperature = 1.0e6 / (kTempTable[index - 1].r * f + kTempTable[index].r * (1.0 - f));
            break;
        }
        last_dt = dt; last_du = du; last_dv = dv;
    }
    (void)last_du; (void)last_dv;
    return temperature;
}

// FindXYZtoCamera, colour-matrix-only (no forward/reduction/camera-
// calibration -- see color_pipeline.h's scope-cut note).
Mat3d find_xyz_to_camera(XY white, const Mat3d& colorMatrix1, const Mat3d& colorMatrix2,
                        double temp1, double temp2) {
    double temperature = temperature_from_xy(white);
    double g;
    if (temperature <= temp1) g = 1.0;
    else if (temperature >= temp2) g = 0.0;
    else g = (1.0 / temperature - 1.0 / temp2) / (1.0 / temp1 - 1.0 / temp2);

    if (g >= 1.0) return colorMatrix1;
    if (g <= 0.0) return colorMatrix2;
    return mat3_add(colorMatrix1, colorMatrix2, g, 1.0 - g);
}

XY neutral_to_xy(Vec3d neutral, const Mat3d& colorMatrix1, const Mat3d& colorMatrix2, double temp1, double temp2) {
    XY last = kD50;
    for (int pass = 0; pass < 30; ++pass) {
        Mat3d xyzToCamera = find_xyz_to_camera(last, colorMatrix1, colorMatrix2, temp1, temp2);
        Mat3d inv = mat3_invert(xyzToCamera);
        Vec3d vec = mat3_apply(inv, neutral);
        XY next = xyz_to_xy(vec);

        if (std::fabs(next.x - last.x) + std::fabs(next.y - last.y) < 0.0000001) return next;
        if (pass == 29) { next.x = (last.x + next.x) * 0.5; next.y = (last.y + next.y) * 0.5; }
        last = next;
    }
    return last;
}

Mat3d mat3_from_row_major9(const float m[9]) {
    Mat3d r;
    for (int i = 0; i < 9; ++i) r.m[i] = (double)m[i];
    return r;
}

}  // namespace

CameraProfile camera_to_pcs_matrix(const DngRaw& raw) {
    Mat3d colorMatrix1 = raw.has_color_matrix1 ? mat3_from_row_major9(raw.color_matrix1)
                                               : Mat3d{1, 0, 0, 0, 1, 0, 0, 0, 1};
    Mat3d colorMatrix2 = raw.has_color_matrix2 ? mat3_from_row_major9(raw.color_matrix2) : Mat3d{};
    double temp1 = illuminant_to_temperature(raw.calibration_illuminant1);
    double temp2 = raw.has_color_matrix2 ? illuminant_to_temperature(raw.calibration_illuminant2) : 0.0;

    if (!raw.has_color_matrix2 || temp1 <= 0.0 || temp2 <= 0.0 || temp1 == temp2) {
        temp1 = temp2 = 5000.0;
        colorMatrix2 = colorMatrix1;
    } else if (temp1 > temp2) {
        std::swap(temp1, temp2);
        std::swap(colorMatrix1, colorMatrix2);
    }

    Vec3d neutral{1, 1, 1};
    if (raw.has_as_shot_neutral) {
        neutral = {raw.as_shot_neutral[0], raw.as_shot_neutral[1], raw.as_shot_neutral[2]};
        double mx = std::max({neutral.x, neutral.y, neutral.z});
        if (mx > 0) neutral = {neutral.x / mx, neutral.y / mx, neutral.z / mx};
    }

    XY white = neutral_to_xy(neutral, colorMatrix1, colorMatrix2, temp1, temp2);

    Mat3d colorMatrix = find_xyz_to_camera(white, colorMatrix1, colorMatrix2, temp1, temp2);

    Vec3d cameraWhiteRaw = mat3_apply(colorMatrix, xy_to_xyz(white));
    double whiteScale = 1.0 / std::max({cameraWhiteRaw.x, cameraWhiteRaw.y, cameraWhiteRaw.z});
    CameraProfile out;
    out.camera_white[0] = (float)pin(0.001, whiteScale * cameraWhiteRaw.x, 1.0);
    out.camera_white[1] = (float)pin(0.001, whiteScale * cameraWhiteRaw.y, 1.0);
    out.camera_white[2] = (float)pin(0.001, whiteScale * cameraWhiteRaw.z, 1.0);

    Mat3d pcsToCamera = mat3_mul(colorMatrix, map_white_matrix(kPCStoXY, white));
    double scale = std::max({mat3_apply(pcsToCamera, kPCStoXYZ).x, mat3_apply(pcsToCamera, kPCStoXYZ).y,
                             mat3_apply(pcsToCamera, kPCStoXYZ).z});
    pcsToCamera = mat3_scale(pcsToCamera, 1.0 / scale);

    out.camera_to_pcs = mat3_invert(pcsToCamera);
    return out;
}

namespace {
Mat3d make_colorspace_to_pcs(Mat3d toPCS) {
    Vec3d w1 = mat3_apply(toPCS, Vec3d{1, 1, 1});
    double s0 = kPCStoXYZ.x / w1.x, s1 = kPCStoXYZ.y / w1.y, s2 = kPCStoXYZ.z / w1.z;
    return mat3_mul(mat3_diag(s0, s1, s2), toPCS);
}
}  // namespace

Mat3d prophoto_to_pcs() {
    return make_colorspace_to_pcs(Mat3d{0.7976749, 0.1351917, 0.0313534,
                                       0.2880402, 0.7118741, 0.0000857,
                                       0.0000000, 0.0000000, 0.8252100});
}
Mat3d prophoto_from_pcs() { return mat3_invert(prophoto_to_pcs()); }

Mat3d srgb50_from_pcs() {
    Mat3d toPCS = make_colorspace_to_pcs(Mat3d{0.4360747, 0.3850649, 0.1430804,
                                               0.2225045, 0.7168786, 0.0606169,
                                               0.0139322, 0.0971045, 0.7141733});
    return mat3_invert(toPCS);
}

namespace {
float apply_srgb_gamma_1_local(float v) {
    if (std::isnan(v)) v = 0.f;
    // GammasRGB (kernel.cu:324-326) clamps to [0,1] before the curve.
    v = std::min(1.0f, std::max(0.0f, v));
    if (v <= 0.0031308f) return 12.92f * v;
    return (1.0f + 0.055f) * std::pow(v, 1.0f / 2.4f) - 0.055f;
}

// ISA's DEFAULT TONE CURVE. SaveAs16BitTiff applies a cubic-interpolated
// 256-entry LUT (Controller.cs:2234-2237, NPP LUTCubic) built by sampling
// the WPF curve editor's spline at i/255 (LUTx, Controller.cs:49-...;
// EvaluateLUT, LUTControl.xaml.cs:399). The editor's constructor seeds a
// pronounced NON-IDENTITY S-curve (LUTControl.xaml.cs:60-71) and the
// control fires its LUT evaluation during startup layout, so this curve IS
// ISA's default render -- an earlier revision assumed the untouched
// default was an identity line, which the source-identity audit disproved
// (the curve lifts ProPhoto ~0.24 to ~0.50). The spline solver below is
// LUTControl's Solve()/Evaluate() (Adobe DNG SDK dng_spline) ported
// verbatim; the per-pixel cubic interpolation through the 256 samples
// models NPP's closed-source nppiLUT_Cubic (an exact-model
// reimplementation, like FilterGaussBorder).
struct DefaultToneLut {
    double y[256];
    DefaultToneLut() {
        static const double px[12] = {
            0.0, 0.012797142375453, 0.0212267133102924, 0.0707504425524739,
            0.12975743909635, 0.237234468515552, 0.323637570597656,
            0.452188527353958, 0.629209516985585, 0.816767470285762,
            0.909492750568996, 1.0};
        static const double pyv[12] = {
            0.0, 0.0105369636685493, 0.0231813200708084, 0.123282474922026,
            0.271853662648571, 0.499452077889235, 0.629056731012391,
            0.766037258703532, 0.887212340891849, 0.962024782938548,
            0.985206103009357, 1.0};
        const int count = 12;
        double S[12];
        // LUTControl.Solve(), verbatim.
        double A = px[1] - px[0];
        double B = (pyv[1] - pyv[0]) / A;
        S[0] = B;
        int j;
        for (j = 2; j < count; ++j) {
            double C = px[j] - px[j - 1];
            double D = (pyv[j] - pyv[j - 1]) / C;
            S[j - 1] = (B * C + D * A) / (A + C);
            A = C;
            B = D;
        }
        S[count - 1] = 2.0 * B - S[count - 2];
        S[0] = 2.0 * S[0] - S[1];
        {
            double E[12], F[12], G[12];
            F[0] = 0.5;
            E[count - 1] = 0.5;
            G[0] = 0.75 * (S[0] + S[1]);
            G[count - 1] = 0.75 * (S[count - 2] + S[count - 1]);
            for (j = 1; j < count - 1; ++j) {
                A = (px[j + 1] - px[j - 1]) * 2.0;
                E[j] = (px[j + 1] - px[j]) / A;
                F[j] = (px[j] - px[j - 1]) / A;
                G[j] = 1.5 * S[j];
            }
            for (j = 1; j < count; ++j) {
                A = 1.0 - F[j - 1] * E[j];
                if (j != count - 1) F[j] /= A;
                G[j] = (G[j] - G[j - 1] * E[j]) / A;
            }
            for (j = count - 2; j >= 0; --j)
                G[j] = G[j] - F[j] * G[j + 1];
            for (j = 0; j < count; ++j) S[j] = G[j];
        }
        // LUTControl.EvaluateSplineSegment / Evaluate, verbatim.
        auto seg = [](double x, double x0, double y0, double s0,
                      double x1, double y1, double s1) {
            const double A_ = x1 - x0;
            const double B_ = (x - x0) / A_;
            const double C_ = (x1 - x) / A_;
            const double D_ = ((y0 * (2.0 - C_ + B_) + (s0 * A_ * B_)) * (C_ * C_)) +
                              ((y1 * (2.0 - B_ + C_) - (s1 * A_ * C_)) * (B_ * B_));
            return std::min(std::max(0.0, D_), 1.0);
        };
        auto evaluate = [&](double x) {
            if (x <= px[0]) return pyv[0];
            if (x >= px[count - 1]) return pyv[count - 1];
            int lower = 1, upper = count - 1;
            while (upper > lower) {
                const int mid = (lower + upper) >> 1;
                if (x == px[mid]) return pyv[mid];
                if (x > px[mid]) lower = mid + 1;
                else upper = mid;
            }
            const int k = lower;
            return seg(x, px[k - 1], pyv[k - 1], S[k - 1], px[k], pyv[k], S[k]);
        };
        // defaultLUT is a float[] in C# -- keep the float quantization.
        for (int i = 0; i < 256; ++i)
            y[i] = (double)(float)evaluate((double)i / 255.0);
    }
    // NPP LUTCubic model: cubic (Catmull-Rom) interpolation through the
    // uniformly spaced samples. Input is pre-clamped to [0,1] by the
    // ThresholdLTGT before the LUT in SaveAs16BitTiff.
    float apply(float v) const {
        const double t = (double)v * 255.0;
        int i = (int)t;
        if (i >= 255) return (float)y[255];
        if (i < 0) return (float)y[0];
        const double f = t - (double)i;
        const double p0 = y[i > 0 ? i - 1 : 0];
        const double p1 = y[i];
        const double p2 = y[i + 1];
        const double p3 = y[(i + 2 <= 255) ? i + 2 : 255];
        const double a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
        const double b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
        const double c = -0.5 * p0 + 0.5 * p2;
        const double r = ((a * f + b) * f + c) * f + p1;
        return (float)std::min(std::max(r, 0.0), 1.0);
    }
};
const DefaultToneLut& default_tone_lut() {
    static const DefaultToneLut lut;
    return lut;
}
}  // namespace

Rgbf render_pixel(Rgbf camera_rgb, const Mat3d& camera_to_prophoto, const Mat3d& prophoto_to_srgb,
                  const float camera_white[3], float exposure_stops) {
    // Clamp to [0, cameraWhite] per channel (highlight protection before
    // the colour-matrix twist, matching SaveAs16BitTiff's own
    // ThresholdLTGT(zeros, zeros, whiteBalance, whiteBalance)).
    float r = std::min(std::max(camera_rgb.r, 0.f), camera_white[0]);
    float g = std::min(std::max(camera_rgb.g, 0.f), camera_white[1]);
    float b = std::min(std::max(camera_rgb.b, 0.f), camera_white[2]);

    Vec3d cam{r, g, b};
    Vec3d pp = mat3_apply(camera_to_prophoto, cam);
    pp = {pin(0.0, pp.x, 1.0), pin(0.0, pp.y, 1.0), pin(0.0, pp.z, 1.0)};

    // ISA's default tone curve (LUTCubic over the seeded S-curve LUT --
    // see DefaultToneLut above), applied between the ProPhoto clamp and
    // the ProPhoto->sRGB twist exactly as in SaveAs16BitTiff.
    const DefaultToneLut& lut = default_tone_lut();
    pp = {(double)lut.apply((float)pp.x), (double)lut.apply((float)pp.y),
          (double)lut.apply((float)pp.z)};

    Vec3d srgb_lin = mat3_apply(prophoto_to_srgb, pp);

    float amp = std::pow(2.0f, exposure_stops);
    Rgbf out;
    out.r = apply_srgb_gamma_1_local((float)srgb_lin.x * amp);
    out.g = apply_srgb_gamma_1_local((float)srgb_lin.y * amp);
    out.b = apply_srgb_gamma_1_local((float)srgb_lin.z * amp);
    return out;
}

Mat3d mat3_mul_public(const Mat3d& A, const Mat3d& B) { return mat3_mul(A, B); }

}  // namespace isacpu
