#include <metal_stdlib>
using namespace metal;

struct BlockMatchParams {
    int ts;
    int search_radius;
};

struct IcaParams {
    int ts;
    int n_iter;
    int img_w;
    int img_h;
};

inline float sample_oob_zero(texture2d<float, access::read> img, int y, int x) {
    if (y >= 0 && y < (int)img.get_height() && x >= 0 && x < (int)img.get_width())
        return img.read(uint2(x, y)).r;
    return 0.0f;
}

inline float bilinear_clamp_edge(texture2d<float, access::read> img, int pixel_y, int pixel_x,
                                   int floor_off_y, int floor_off_x, float frac_x, float frac_y) {
    int floor_y = clamp(pixel_y + floor_off_y, 0, (int)img.get_height() - 1);
    int floor_x = clamp(pixel_x + floor_off_x, 0, (int)img.get_width() - 1);
    int ceil_y = clamp(floor_y + 1, 0, (int)img.get_height() - 1);
    int ceil_x = clamp(floor_x + 1, 0, (int)img.get_width() - 1);

    float m00 = img.read(uint2(floor_x, floor_y)).r;
    float m01 = img.read(uint2(ceil_x, floor_y)).r;
    float m10 = img.read(uint2(floor_x, ceil_y)).r;
    float m11 = img.read(uint2(ceil_x, ceil_y)).r;

    float lerpx_top = m00 + (m01 - m00) * frac_x;
    float lerpx_bot = m10 + (m11 - m10) * frac_x;
    return lerpx_top + (lerpx_bot - lerpx_top) * frac_y;
}

inline float bilinear_oob_zero(texture2d<float, access::read> img, int ts, int pixel_y, int pixel_x,
                               int floor_off_y, int floor_off_x, float frac_x, float frac_y) {
    int floor_y = pixel_y + floor_off_y;
    int floor_x = pixel_x + floor_off_x;
    float m00 = sample_oob_zero(img, floor_y, floor_x);
    float m01 = sample_oob_zero(img, floor_y, floor_x + 1);
    float m10 = sample_oob_zero(img, floor_y + 1, floor_x);
    float m11 = sample_oob_zero(img, floor_y + 1, floor_x + 1);
    float lerpx_top = m00 + (m01 - m00) * frac_x;
    float lerpx_bot = m10 + (m11 - m10) * frac_x;
    return lerpx_top + (lerpx_bot - lerpx_top) * frac_y;
}

kernel void kernel_compute_sobel(
    texture2d<float, access::read> inTexture [[texture(0)]],
    texture2d<float, access::write> gradxTex [[texture(1)]],
    texture2d<float, access::write> gradyTex [[texture(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) return;

    int w = (int)inTexture.get_width();
    int h = (int)inTexture.get_height();

    float mx = (gid.x > 0) ? inTexture.read(uint2(gid.x - 1, gid.y)).r : 0.0f;
    float px = ((int)gid.x + 1 < w) ? inTexture.read(uint2(gid.x + 1, gid.y)).r : 0.0f;
    float gx = px - mx;

    float my = (gid.y > 0) ? inTexture.read(uint2(gid.x, gid.y - 1)).r : 0.0f;
    float py = ((int)gid.y + 1 < h) ? inTexture.read(uint2(gid.x, gid.y + 1)).r : 0.0f;
    float gy = py - my;

    gradxTex.write(float4(gx, 0, 0, 1), gid);
    gradyTex.write(float4(gy, 0, 0, 1), gid);
}

kernel void kernel_block_match_L1(
    texture2d<float, access::read> refTex [[texture(0)]],
    texture2d<float, access::read> movTex [[texture(1)]],
    texture2d<float, access::read> inFlow [[texture(2)]],
    texture2d<float, access::write> outFlow [[texture(3)]],
    constant BlockMatchParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outFlow.get_width() || gid.y >= outFlow.get_height()) return;

    int tx = (int)gid.x;
    int ty = (int)gid.y;
    int ts = params.ts;
    int R = params.search_radius;

    float2 initial_flow = inFlow.read(gid).rg;
    int flow_dx = (int)rint(initial_flow.x);
    int flow_dy = (int)rint(initial_flow.y);

    int ox = tx * ts;
    int oy = ty * ts;

    float best_dist = 1e38f;
    int best_dy = 0;
    int best_dx = 0;

    for (int s_dy = -R; s_dy <= R; ++s_dy) {
        for (int s_dx = -R; s_dx <= R; ++s_dx) {
            float dist = 0.0f;
            for (int i = 0; i < ts; ++i) {
                int ry = oy + i;
                if (ry >= (int)refTex.get_height()) break;
                for (int j = 0; j < ts; ++j) {
                    int rx = ox + j;
                    if (rx >= (int)refTex.get_width()) break;

                    int my = ry + flow_dy + s_dy;
                    int mx = rx + flow_dx + s_dx;
                    float m_val = 0.0f;
                    if (my >= 0 && my < (int)movTex.get_height() &&
                        mx >= 0 && mx < (int)movTex.get_width())
                        m_val = movTex.read(uint2(mx, my)).r;

                    dist += abs(refTex.read(uint2(rx, ry)).r - m_val);
                }
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_dy = s_dy;
                best_dx = s_dx;
            }
        }
    }

    outFlow.write(float4((float)(flow_dx + best_dx), (float)(flow_dy + best_dy), 0, 1), gid);
}

kernel void kernel_ica_refine(
    texture2d<float, access::read> refTex [[texture(0)]],
    texture2d<float, access::read> gxTex [[texture(1)]],
    texture2d<float, access::read> gyTex [[texture(2)]],
    texture2d<float, access::read> movTex [[texture(3)]],
    texture2d<float, access::read> inFlow [[texture(4)]],
    texture2d<float, access::write> outFlow [[texture(5)]],
    constant float* hessInv [[buffer(0)]],
    constant IcaParams& params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= outFlow.get_width() || gid.y >= outFlow.get_height()) return;

    int tx = (int)gid.x;
    int ty = (int)gid.y;
    int ts = params.ts;
    int nx = (int)outFlow.get_width();

    float2 flow = inFlow.read(gid).rg;
    size_t hbase = ((size_t)ty * nx + tx) * 4;
    float valid = hessInv[hbase + 3];
    if (valid < 0.5f) {
        outFlow.write(float4(flow.x, flow.y, 0, 1), gid);
        return;
    }

    float ih11 = hessInv[hbase + 0];
    float ih12 = hessInv[hbase + 1];
    float ih22 = hessInv[hbase + 2];

    int ox = tx * ts;
    int oy = ty * ts;

    for (int iter = 0; iter < params.n_iter; ++iter) {
        float frac_x = flow.x - trunc(flow.x);
        int floor_off_x = (int)trunc(flow.x);
        float frac_y = flow.y - trunc(flow.y);
        int floor_off_y = (int)trunc(flow.y);

        float B0 = 0.0f;
        float B1 = 0.0f;

        for (int i = 0; i < ts; ++i) {
            int py = oy + i;
            if (py >= params.img_h) break;
            for (int j = 0; j < ts; ++j) {
                int px = ox + j;
                if (px >= params.img_w) break;

                float mov_interp = (ts == 8)
                    ? bilinear_clamp_edge(movTex, py, px, floor_off_y, floor_off_x, frac_x, frac_y)
                    : bilinear_oob_zero(movTex, ts, py, px, floor_off_y, floor_off_x, frac_x, frac_y);

                float gradt = mov_interp - refTex.read(uint2(px, py)).r;
                float gx = gxTex.read(uint2(px, py)).r;
                float gy = gyTex.read(uint2(px, py)).r;

                B0 += -gx * gradt;
                B1 += -gy * gradt;
            }
        }

        flow.x += ih11 * B0 + ih12 * B1;
        flow.y += ih12 * B0 + ih22 * B1;
    }

    outFlow.write(float4(flow.x, flow.y, 0, 1), gid);
}
