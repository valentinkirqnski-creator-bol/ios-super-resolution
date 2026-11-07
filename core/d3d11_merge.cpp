#ifdef _WIN32
#include "d3d11_merge.h"

#include <d3d11.h>
#include <d3dcompiler.h>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace hhsr {
namespace {

// ---- constant buffer (must match cbuffer Params in the HLSL below) -------
struct Params {
    uint32_t band_h, Ws, y0, lr_h;
    uint32_t lr_w, flow_ny, flow_nx, covs_h;
    uint32_t covs_w, rob_h, rob_w, rnc;
    uint32_t tile_size, iso, bayer, use_homography;
    uint32_t rob_is_output, rob_is_raw, rob_bilinear, nch;
    float    scale; uint32_t cfa0, cfa1, cfa2;
    uint32_t cfa3; float pad0, pad1, pad2;
    float    hmat0, hmat1, hmat2, hmat3;
    float    hmat4, hmat5, hmat6, hmat7;
    float    hmat8; float pad3, pad4, pad5;
};
static_assert(sizeof(Params) == 160, "Params must be 16-byte-row aligned");

// Faithful translation of accumulate_comp + interp_inv_cov + soften_inv_cov
// (merge.cpp). raw_det covariance path only (comp frames).
static const char* kMergeHLSL = R"HLSL(
cbuffer Params : register(b0) {
    uint band_h; uint Ws; uint y0; uint lr_h;
    uint lr_w; uint flow_ny; uint flow_nx; uint covs_h;
    uint covs_w; uint rob_h; uint rob_w; uint rnc;
    uint tile_size; uint iso; uint bayer; uint use_homography;
    uint rob_is_output; uint rob_is_raw; uint rob_bilinear; uint nch;
    float scale; uint cfa0; uint cfa1; uint cfa2;
    uint cfa3; float pad0; float pad1; float pad2;
    float hmat0; float hmat1; float hmat2; float hmat3;
    float hmat4; float hmat5; float hmat6; float hmat7;
    float hmat8; float pad3; float pad4; float pad5;
};
StructuredBuffer<float>   img  : register(t0);
StructuredBuffer<float>   flow : register(t1);
StructuredBuffer<float>   covs : register(t2);
StructuredBuffer<float>   rob  : register(t3);
RWStructuredBuffer<float> numb : register(u0);
RWStructuredBuffer<float> denb : register(u1);

int rnd(float x){ return (int)floor(x + 0.5f); }  // == std::lround for x >= -0.5

float rob_nearest(float ry, float rx, uint ch){
    int iy = clamp((int)floor(ry + 0.5f), 0, (int)rob_h - 1);
    int ix = clamp((int)floor(rx + 0.5f), 0, (int)rob_w - 1);
    uint c = min(ch, rnc - 1u);
    return rob[((uint)iy * rob_w + (uint)ix) * rnc + c];
}
float rob_bilin(float ry, float rx, uint ch){
    ry = clamp(ry, 0.0f, (float)rob_h - 1.0f);
    rx = clamp(rx, 0.0f, (float)rob_w - 1.0f);
    int y0i = (int)floor(ry); int x0i = (int)floor(rx);
    int y1i = min(y0i + 1, (int)rob_h - 1); int x1i = min(x0i + 1, (int)rob_w - 1);
    float fy = ry - (float)y0i; float fx = rx - (float)x0i;
    uint c = min(ch, rnc - 1u);
    float v00 = rob[((uint)y0i * rob_w + (uint)x0i) * rnc + c];
    float v01 = rob[((uint)y0i * rob_w + (uint)x1i) * rnc + c];
    float v10 = rob[((uint)y1i * rob_w + (uint)x0i) * rnc + c];
    float v11 = rob[((uint)y1i * rob_w + (uint)x1i) * rnc + c];
    float top = v00 + (v01 - v00) * fx;
    float bot = v10 + (v11 - v10) * fx;
    return top + (bot - top) * fy;
}

[numthreads(16,16,1)]
void main(uint3 gid : SV_DispatchThreadID){
    if (gid.x >= Ws || gid.y >= band_h) return;
    uint hr_j = gid.x; uint local_i = gid.y;
    int  hr_i = (int)y0 + (int)local_i;
    float lr_x = (float)hr_j / scale;
    float lr_y = (float)hr_i / scale;

    int px = (int)(lr_x / (float)tile_size);
    int py = (int)(lr_y / (float)tile_size);
    px = clamp(px, 0, (int)flow_nx - 1);
    py = clamp(py, 0, (int)flow_ny - 1);
    float flowx = flow[((uint)py * flow_nx + (uint)px) * 2u + 0u];
    float flowy = flow[((uint)py * flow_nx + (uint)px) * 2u + 1u];

    // robustness fetch position (matches merge.cpp accumulate_comp)
    float rob_y, rob_x;
    if (rob_is_output != 0u) { rob_y = (float)hr_i; rob_x = (float)hr_j; }
    else {
        rob_y = lr_y; rob_x = lr_x;
        if (rob_is_raw == 0u && bayer != 0u) { rob_y = (lr_y - 0.5f) / 2.0f; rob_x = (lr_x - 0.5f) / 2.0f; }
    }
    float lr0, lr1, lr2;
    if (rob_bilinear != 0u) { lr0 = rob_bilin(rob_y, rob_x, 0u); lr1 = rob_bilin(rob_y, rob_x, 1u); lr2 = rob_bilin(rob_y, rob_x, 2u); }
    else { lr0 = rob_nearest(rob_y, rob_x, 0u); lr1 = rob_nearest(rob_y, rob_x, 1u); lr2 = rob_nearest(rob_y, rob_x, 2u); }
    if (lr0 <= 0.0f && lr1 <= 0.0f && lr2 <= 0.0f) return;
    float lrc[3]; lrc[0] = lr0; lrc[1] = lr1; lrc[2] = lr2;

    float lr_mov_x = lr_x + flowx;
    float lr_mov_y = lr_y + flowy;
    if (use_homography != 0u) {
        float wq = hmat6 * lr_mov_x + hmat7 * lr_mov_y + hmat8;
        float iwq = (wq != 0.0f) ? 1.0f / wq : 0.0f;
        float hx = (hmat0 * lr_mov_x + hmat1 * lr_mov_y + hmat2) * iwq;
        float hy = (hmat3 * lr_mov_x + hmat4 * lr_mov_y + hmat5) * iwq;
        lr_mov_x = hx; lr_mov_y = hy;
    }
    if (!(lr_mov_x >= 0.0f && lr_mov_x < (float)lr_w && lr_mov_y >= 0.0f && lr_mov_y < (float)lr_h)) return;

    float ixx = 0.0f, ixy = 0.0f, iyy = 0.0f;
    if (iso == 0u) {
        float kmap_j, kmap_i;
        if (bayer != 0u) { kmap_j = (lr_mov_x - 0.5f) / 2.0f; kmap_i = (lr_mov_y - 0.5f) / 2.0f; }
        else { kmap_j = lr_mov_x - 0.5f; kmap_i = lr_mov_y - 0.5f; }
        float frac_x = kmap_j - trunc(kmap_j);
        float frac_y = kmap_i - trunc(kmap_i);
        int fx = max((int)kmap_j, 0);           // int() truncates toward zero
        int fy = max((int)kmap_i, 0);
        int cx = min(fx + 1, (int)covs_w - 1);
        int cy = min(fy + 1, (int)covs_h - 1);
        uint tl = ((uint)fy * covs_w + (uint)fx) * 4u;
        uint tr = ((uint)fy * covs_w + (uint)cx) * 4u;
        uint bl = ((uint)cy * covs_w + (uint)fx) * 4u;
        uint br = ((uint)cy * covs_w + (uint)cx) * 4u;
        float xx = (covs[tl+0] + frac_x*(covs[tr+0]-covs[tl+0])) + frac_y*((covs[bl+0]+frac_x*(covs[br+0]-covs[bl+0])) - (covs[tl+0]+frac_x*(covs[tr+0]-covs[tl+0])));
        float xy = (covs[tl+1] + frac_x*(covs[tr+1]-covs[tl+1])) + frac_y*((covs[bl+1]+frac_x*(covs[br+1]-covs[bl+1])) - (covs[tl+1]+frac_x*(covs[tr+1]-covs[tl+1])));
        float yy = (covs[tl+3] + frac_x*(covs[tr+3]-covs[tl+3])) + frac_y*((covs[bl+3]+frac_x*(covs[br+3]-covs[bl+3])) - (covs[tl+3]+frac_x*(covs[tr+3]-covs[tl+3])));
        float det = xx * yy - xy * xy;
        if (abs(det) > 1e-10f) { float inv = 1.0f / det; ixx = inv*yy; ixy = -inv*xy; iyy = inv*xx; }
        else { ixx = 1.0f; ixy = 0.0f; iyy = 1.0f; }
        // soften_inv_cov
        float m = max(abs(ixx), max(abs(iyy), abs(ixy)));
        if (m > 32.0f && isfinite(m)) { float s = 32.0f / m; ixx *= s; ixy *= s; iyy *= s; }
        else if (!(isfinite(ixx) && isfinite(ixy) && isfinite(iyy))) { ixx = 2.0f; ixy = 0.0f; iyy = 2.0f; }
    }

    int center_j = rnd(lr_mov_x);
    int center_i = rnd(lr_mov_y);
    float val[3]; float acc[3];
    val[0]=val[1]=val[2]=0.0f; acc[0]=acc[1]=acc[2]=0.0f;
    uint cfa[4]; cfa[0]=cfa0; cfa[1]=cfa1; cfa[2]=cfa2; cfa[3]=cfa3;
    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            int j = center_j + dj; int i = center_i + di;
            if (!(j >= 0 && j < (int)lr_w && i >= 0 && i < (int)lr_h)) continue;
            uint ch = (bayer != 0u) ? cfa[(uint)((i & 1) * 2 + (j & 1))] : 0u;
            float c = img[(uint)i * lr_w + (uint)j];
            float dx = (float)j - lr_mov_x;
            float dy = (float)i - lr_mov_y;
            float z = (iso != 0u) ? 2.0f * (dx*dx + dy*dy)
                                  : ixx*dx*dx + 2.0f*ixy*dx*dy + iyy*dy*dy;
            z = max(0.0f, z);
            float w = exp(-0.5f * z);
            float lc = lrc[(ch < 3u) ? ch : 2u];
            val[ch] += w * lc * c;
            acc[ch] += w * lc;
        }
    }
    uint base = (local_i * Ws + hr_j) * nch;
    for (uint ch = 0u; ch < nch; ++ch) { numb[base + ch] += val[ch]; denb[base + ch] += acc[ch]; }
}
)HLSL";

// ---- device + shader (lazy singletons) -----------------------------------
struct FrameBufs {
    ID3D11Buffer* img = nullptr; ID3D11ShaderResourceView* imgSRV = nullptr;
    ID3D11Buffer* flow = nullptr; ID3D11ShaderResourceView* flowSRV = nullptr;
    ID3D11Buffer* covs = nullptr; ID3D11ShaderResourceView* covsSRV = nullptr;
    ID3D11Buffer* rob = nullptr; ID3D11ShaderResourceView* robSRV = nullptr;
    const float* img_ptr = nullptr;  // staleness check
    void release() {
        if (imgSRV) imgSRV->Release(); if (img) img->Release();
        if (flowSRV) flowSRV->Release(); if (flow) flow->Release();
        if (covsSRV) covsSRV->Release(); if (covs) covs->Release();
        if (robSRV) robSRV->Release(); if (rob) rob->Release();
        *this = FrameBufs{};
    }
};

struct Ctx {
    ID3D11Device* dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    ID3D11ComputeShader* cs = nullptr;
    ID3D11Buffer* cb = nullptr;
    bool init_tried = false, init_ok = false;
    std::unordered_map<int, FrameBufs> frames;
};
static Ctx g;

static bool ensure_init() {
    if (g.init_tried) return g.init_ok;
    g.init_tried = true;
    D3D_FEATURE_LEVEL fl;
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                                   D3D11_SDK_VERSION, &g.dev, &fl, &g.ctx);
    if (FAILED(hr) || !g.dev) return false;
    ID3DBlob* code = nullptr; ID3DBlob* err = nullptr;
    hr = D3DCompile(kMergeHLSL, std::strlen(kMergeHLSL), "merge", nullptr, nullptr,
                    "main", "cs_5_0", 0, 0, &code, &err);
    if (err) err->Release();
    if (FAILED(hr) || !code) return false;
    hr = g.dev->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &g.cs);
    code->Release();
    if (FAILED(hr) || !g.cs) return false;
    D3D11_BUFFER_DESC cbd{};
    cbd.ByteWidth = sizeof(Params);
    cbd.Usage = D3D11_USAGE_DEFAULT;
    cbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    if (FAILED(g.dev->CreateBuffer(&cbd, nullptr, &g.cb)) || !g.cb) return false;
    g.init_ok = true;
    return true;
}

// Create an immutable StructuredBuffer<float> + its SRV from host data.
static bool make_sb(const float* data, UINT nfloats, ID3D11Buffer** buf, ID3D11ShaderResourceView** srv) {
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = nfloats * 4u;
    bd.Usage = D3D11_USAGE_IMMUTABLE;
    bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = 4u;
    D3D11_SUBRESOURCE_DATA sd{ data, 0, 0 };
    if (FAILED(g.dev->CreateBuffer(&bd, &sd, buf)) || !*buf) return false;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = DXGI_FORMAT_UNKNOWN;
    sv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    sv.Buffer.NumElements = nfloats;
    if (FAILED(g.dev->CreateShaderResourceView(*buf, &sv, srv)) || !*srv) return false;
    return true;
}

// Create a default RWStructuredBuffer<float> + UAV, initialised from host data.
static bool make_rw(const float* data, UINT nfloats, ID3D11Buffer** buf, ID3D11UnorderedAccessView** uav) {
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = nfloats * 4u;
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.StructureByteStride = 4u;
    D3D11_SUBRESOURCE_DATA sd{ data, 0, 0 };
    if (FAILED(g.dev->CreateBuffer(&bd, &sd, buf)) || !*buf) return false;
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};
    ud.Format = DXGI_FORMAT_UNKNOWN;
    ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    ud.Buffer.NumElements = nfloats;
    if (FAILED(g.dev->CreateUnorderedAccessView(*buf, &ud, uav)) || !*uav) return false;
    return true;
}

// Read a default buffer back into host memory via a staging copy.
static bool readback(ID3D11Buffer* src, UINT nfloats, float* out) {
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth = nfloats * 4u;
    bd.Usage = D3D11_USAGE_STAGING;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Buffer* stg = nullptr;
    if (FAILED(g.dev->CreateBuffer(&bd, nullptr, &stg)) || !stg) return false;
    g.ctx->CopyResource(stg, src);
    D3D11_MAPPED_SUBRESOURCE ms{};
    bool ok = SUCCEEDED(g.ctx->Map(stg, 0, D3D11_MAP_READ, 0, &ms));
    if (ok) { std::memcpy(out, ms.pData, nfloats * 4u); g.ctx->Unmap(stg, 0); }
    stg->Release();
    return ok;
}

}  // namespace

bool d3d11_merge_available() { return ensure_init(); }

void d3d11_merge_reset() {
    for (auto& kv : g.frames) kv.second.release();
    g.frames.clear();
}

bool merge_comp_band_d3d11(const Image& comp_raw, const FlowField& flow, const CovField& covs,
                           const Image& robustness, int tile_size,
                           Image& num_band, Image& den_band, int y0, const Config& cfg,
                           int frame_id) {
    // Unsupported configurations -> let the caller run the CPU accumulate.
    if (flow.has_dense() || (cfg.affine_flow_enabled && flow.has_affine())) return false;
    if (cfg.gcam_flow_bilinear_active()) return false;   // bilinear flow not ported -> CPU
    if (robustness.c > 1) return false;                 // per-channel R not ported yet
    if (comp_raw.h <= 0 || comp_raw.w <= 0) return false;  // no resident raw pixels
    if (robustness.h <= 0 || robustness.w <= 0) return false;
    if (!ensure_init()) return false;

    const int band_h = num_band.h, Ws = num_band.w;
    const int lr_h = comp_raw.h, lr_w = comp_raw.w;
    const int nch = cfg.bayer_mode ? 3 : 1;
    if (num_band.c != nch || den_band.c != nch) return false;

    const long long out_h = llround((double)lr_h * (double)cfg.scale);
    const long long out_w = llround((double)lr_w * (double)cfg.scale);
    const bool rob_is_output = (robustness.h == out_h && robustness.w == out_w);
    const bool rob_is_raw = (robustness.h == lr_h && robustness.w == lr_w);
    const bool rob_bilinear = !rob_is_output &&
        (cfg.merge_robustness_bilinear || cfg.robustness_guide_res_bilinear ||
         (cfg.gcam_sabre_bilinear_upscale && !cfg.gcam_sabre_raw_res));

    // Per-frame input buffers (cached across bands by frame_id).
    FrameBufs* fb = nullptr;
    if (frame_id >= 0) {
        fb = &g.frames[frame_id];
        if (fb->img && fb->img_ptr != comp_raw.data.data()) { fb->release(); }  // stale
    }
    FrameBufs local;
    FrameBufs* use = fb ? fb : &local;
    bool ok = true;
    if (!use->img) {
        ok = ok && make_sb(comp_raw.data.data(), (UINT)comp_raw.data.size(), &use->img, &use->imgSRV);
        ok = ok && make_sb(flow.flow.data(), (UINT)flow.flow.size(), &use->flow, &use->flowSRV);
        ok = ok && make_sb(covs.cov.data(), (UINT)covs.cov.size(), &use->covs, &use->covsSRV);
        ok = ok && make_sb(robustness.data.data(), (UINT)robustness.data.size(), &use->rob, &use->robSRV);
        use->img_ptr = comp_raw.data.data();
        if (!ok) { use->release(); if (fb) g.frames.erase(frame_id); return false; }
    }

    // num/den: upload current band contents, accumulate on GPU, read back.
    ID3D11Buffer* numBuf = nullptr; ID3D11UnorderedAccessView* numUAV = nullptr;
    ID3D11Buffer* denBuf = nullptr; ID3D11UnorderedAccessView* denUAV = nullptr;
    const UINT ndn = (UINT)num_band.data.size();
    if (!make_rw(num_band.data.data(), ndn, &numBuf, &numUAV) ||
        !make_rw(den_band.data.data(), ndn, &denBuf, &denUAV)) {
        if (numUAV) numUAV->Release(); if (numBuf) numBuf->Release();
        if (denUAV) denUAV->Release(); if (denBuf) denBuf->Release();
        if (!fb) use->release();
        return false;
    }

    Params p{};
    p.band_h = band_h; p.Ws = Ws; p.y0 = (uint32_t)y0; p.lr_h = lr_h;
    p.lr_w = lr_w; p.flow_ny = flow.ny; p.flow_nx = flow.nx; p.covs_h = covs.h;
    p.covs_w = covs.w; p.rob_h = robustness.h; p.rob_w = robustness.w; p.rnc = (uint32_t)robustness.c;
    p.tile_size = (uint32_t)tile_size;
    p.iso = (cfg.kernel == KernelShape::Iso) ? 1u : 0u;
    p.bayer = cfg.bayer_mode ? 1u : 0u;
    p.use_homography = flow.has_global_h ? 1u : 0u;
    p.rob_is_output = rob_is_output ? 1u : 0u;
    p.rob_is_raw = rob_is_raw ? 1u : 0u;
    p.rob_bilinear = rob_bilinear ? 1u : 0u;
    p.nch = (uint32_t)nch;
    p.scale = cfg.scale;
    p.cfa0 = (uint32_t)cfg.cfa.p[0][0]; p.cfa1 = (uint32_t)cfg.cfa.p[0][1];
    p.cfa2 = (uint32_t)cfg.cfa.p[1][0]; p.cfa3 = (uint32_t)cfg.cfa.p[1][1];
    const float* h = flow.global_h;
    p.hmat0 = h[0]; p.hmat1 = h[1]; p.hmat2 = h[2];
    p.hmat3 = h[3]; p.hmat4 = h[4]; p.hmat5 = h[5];
    p.hmat6 = h[6]; p.hmat7 = h[7]; p.hmat8 = h[8];
    g.ctx->UpdateSubresource(g.cb, 0, nullptr, &p, 0, 0);

    g.ctx->CSSetShader(g.cs, nullptr, 0);
    ID3D11ShaderResourceView* srvs[4] = { use->imgSRV, use->flowSRV, use->covsSRV, use->robSRV };
    g.ctx->CSSetShaderResources(0, 4, srvs);
    ID3D11UnorderedAccessView* uavs[2] = { numUAV, denUAV };
    UINT initc[2] = { 0, 0 };
    g.ctx->CSSetUnorderedAccessViews(0, 2, uavs, initc);
    g.ctx->CSSetConstantBuffers(0, 1, &g.cb);
    g.ctx->Dispatch((UINT)((Ws + 15) / 16), (UINT)((band_h + 15) / 16), 1);

    ok = readback(numBuf, ndn, num_band.data.data()) &&
         readback(denBuf, ndn, den_band.data.data());

    // Unbind UAVs so the buffers can be released cleanly.
    ID3D11UnorderedAccessView* nulluav[2] = { nullptr, nullptr };
    g.ctx->CSSetUnorderedAccessViews(0, 2, nulluav, initc);
    numUAV->Release(); numBuf->Release(); denUAV->Release(); denBuf->Release();
    if (!fb) use->release();
    return ok;
}

}  // namespace hhsr
#endif  // _WIN32
